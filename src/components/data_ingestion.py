import os
from datasets import load_dataset
from dotenv import load_dotenv

class DataIngestion:
    def __init__(self):
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        self.dataset_name = "samsum"
        self.raw_data_path = "artifacts/data/raw"

    def download_data(self):
        print(f"Downloading {self.dataset_name} dataset...")
        # Authenticate if a token is provided in .env
        token_arg = self.hf_token if self.hf_token and self.hf_token != "hf_your_token_here" else None
        
        dataset = load_dataset(self.dataset_name, token=token_arg, trust_remote_code=True)
        os.makedirs(self.raw_data_path, exist_ok=True)
        dataset.save_to_disk(self.raw_data_path)
        print(f"Dataset successfully saved to {self.raw_data_path}")
        return self.raw_data_path
