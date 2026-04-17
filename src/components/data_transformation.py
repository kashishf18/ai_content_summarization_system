import os
from datasets import load_from_disk
from transformers import AutoTokenizer

class DataTransformation:
    def __init__(self):
        self.raw_data_path = "artifacts/data/raw"
        self.processed_data_path = "artifacts/data/processed"
        self.tokenizer_name = "t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # T5 requires a prefix for the task
        inputs = ["summarize: " + doc for doc in example_batch["dialogue"]]
        input_encodings = self.tokenizer(inputs, max_length=1024, truncation=True)
        target_encodings = self.tokenizer(text_target=example_batch['summary'], max_length=128, truncation=True)
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def process(self):
        print("Starting data transformation...")
        dataset = load_from_disk(self.raw_data_path)
        
        # Apply transformation in batches
        dataset_pt = dataset.map(self.convert_examples_to_features, batched=True)
        
        # Remove original plain text columns to save space if needed
        columns_to_remove = ["id", "dialogue", "summary"]
        dataset_pt = dataset_pt.remove_columns(columns_to_remove)
        
        # Save tokenized processed datasets
        os.makedirs(self.processed_data_path, exist_ok=True)
        dataset_pt.save_to_disk(self.processed_data_path)
        print(f"Transformed dataset successfully saved to {self.processed_data_path}")
