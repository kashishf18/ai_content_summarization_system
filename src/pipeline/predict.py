import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class PredictionPipeline:
    def __init__(self, model_path=None):
        self.model_path = model_path or "artifacts/model/t5-samsum-model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Warning: Trained artifacts missing. Defaulting to base model: t5-small")
            self.model_path = "t5-small"
            
        print(f"Loading tokenizer and model={self.model_path} on device={self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

    def predict(self, text: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and Tokenizer not loaded. Call load_model() first.")
        # T5 requires the task prefix
        text = "summarize: " + text
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
        
        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summary
