import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate

class ModelEvaluation:
    def __init__(self):
        self.model_path = "artifacts/model/t5-samsum-model"
        self.tokenizer_path = "artifacts/model/t5-samsum-model"
        self.raw_data_path = "artifacts/data/raw"
        self.metrics_folder = "artifacts/metrics"
        self.metrics_file = "artifacts/metrics/rouge_scores.json"

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, batch_size=16, device="cuda"):
        # T5 requires the task prefix
        prefixed_dialogues = ["summarize: " + doc for doc in dataset["dialogue"]]
        article_batches = list(self.generate_batch_sized_chunks(prefixed_dialogues, batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset["summary"], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
            
            # Predict summaries
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device), 
                length_penalty=0.8, num_beams=8, max_length=128
            )
            
            # Decode the generated summaries into text
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        # Get final score
        score = metric.compute()
        return score

    def evaluate(self):
        print("Starting model evaluation...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(device)
        
        if not os.path.exists(self.raw_data_path):
            print("Raw data not found! Pipeline needs raw text for decoding.")
            return

        dataset_samsum_raw = load_from_disk(self.raw_data_path)
        rouge_metric = evaluate.load("rouge")
        
        # Taking a tiny sample to calculate quick ROUGE scores for AWS Free Tier restrictions (~10 outputs limits)
        test_data = dataset_samsum_raw['test'].select(range(min(10, len(dataset_samsum_raw['test']))))
        
        score = self.calculate_metric_on_test_ds(
            test_data, rouge_metric, model, tokenizer, batch_size=2, device=device
        )
        
        # Standardize ROUGE metric format
        rouge_dict = {
            "rouge1": score["rouge1"],
            "rouge2": score["rouge2"],
            "rougeL": score["rougeL"],
            "rougeLsum": score["rougeLsum"],
        }
        
        os.makedirs(self.metrics_folder, exist_ok=True)
        with open(self.metrics_file, "w") as f:
            json.dump(rouge_dict, f, indent=4)
            
        print(f"Metrics saved at {self.metrics_file}")
