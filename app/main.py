from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import sys
import os

# Root imports fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.schemas import PredictRequest, PredictResponse
from src.pipeline.predict import PredictionPipeline
from src.pipeline.stage_02_train import ModelTrainingPipeline
from src.pipeline.stage_01_data import DataPipeline

pipeline_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Startup Initialization...")
    model_path = os.getenv("ARTIFACT_MODEL_PATH", "artifacts/model/t5-samsum-model")
    
    # Artifact existence check
    if not os.path.exists(model_path):
        print(f"WARNING: Trained model folder '{model_path}' is absent. Prediction will fallback to t5-small base model.")
    
    try:
        pipeline = PredictionPipeline(model_path=model_path)
        pipeline.load_model()
        app.state.pipeline = pipeline
        print("Model Pipeline Loaded successfully and stored in app.state!")
    except Exception as e:
        print(f"Failed to load pipeline on startup: {e}")
        app.state.pipeline = None
    yield
    print("Shutting down... freeing GPU/CPU payload.")
    app.state.pipeline = None

app = FastAPI(title="AI Summarization Pipeline API", version="1.0.0", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def index_route():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AI Summarizer API is running</h1><p>Index file not found.</p>"

@app.get("/health")
def health_check():
    status = "ready" if getattr(app.state, "pipeline", None) else "loading/error"
    return {"status": status, "message": "The AI is up and running!"}

def run_training_pipeline_background():
    try:
        DataPipeline().run()
        ModelTrainingPipeline().run()
        
        # Reload pipeline into state after training
        pipeline = PredictionPipeline()
        pipeline.load_model()
        app.state.pipeline = pipeline
        print("Background tuning successful. Production pipeline swapped into state.")
    except Exception as e:
        print(f"Background tuning failed: {e}")

@app.get("/train")
def train_route(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_pipeline_background)
    return {"message": "Background fine-tuning initiated successfully. Monitor terminal process."}

@app.post("/predict", response_model=PredictResponse)
def predict_route(request: PredictRequest):
    pipeline = getattr(app.state, "pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="The summarization payload is caching or failed to load. Please wait.")
    
    try:
        generated_summary = pipeline.predict(request.text)
        return PredictResponse(summary=generated_summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
