import sys
import os

# Allow import from local root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        print("====== Stage 02: Model Training Started ======")
        
        trainer = ModelTrainer()
        trainer.train()
        print("Model training completed and artifacts dumped.")

        evaluator = ModelEvaluation()
        evaluator.evaluate()
        print("Model evaluation completed and verified.")
        
        print("====== Stage 02: Model Training End ======")

if __name__ == '__main__':
    pipeline = ModelTrainingPipeline()
    pipeline.run()
