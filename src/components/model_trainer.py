import os
import yaml
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch

class ModelTrainer:
    def __init__(self):
        self.params_yaml = "params.yaml"
        self.processed_data_path = "artifacts/data/processed"
        self.model_ckpt = "t5-small"
        self.output_model_path = "artifacts/model/t5-samsum-model"

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {self.model_ckpt} and tokenizer on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # Load processed datasets prepared in stage 1
        dataset_samsum_pt = load_from_disk(self.processed_data_path)

        # Load parameters from yaml config
        with open(self.params_yaml, "r") as f:
            params = yaml.safe_load(f)
            train_params = params.get("TrainingArguments", {})
            
        trainer_args = Seq2SeqTrainingArguments(
            output_dir=self.output_model_path,
            num_train_epochs=train_params.get("num_train_epochs", 1),
            warmup_steps=train_params.get("warmup_steps", 500),
            per_device_train_batch_size=train_params.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=train_params.get("per_device_eval_batch_size", 1),
            weight_decay=train_params.get("weight_decay", 0.01),
            logging_steps=train_params.get("logging_steps", 10),
            eval_strategy=train_params.get("evaluation_strategy", "steps"),
            eval_steps=train_params.get("eval_steps", 500),
            save_steps=float(train_params.get("save_steps", 1e6)),
            gradient_accumulation_steps=train_params.get("gradient_accumulation_steps", 16)
        ) 

        # We limit the training set specifically to keep inference/testing fast & lightweight on Free Tier VMs length=50
        train_data = dataset_samsum_pt["train"].select(range(min(50, len(dataset_samsum_pt["train"]))))
        eval_data = dataset_samsum_pt["validation"].select(range(min(10, len(dataset_samsum_pt["validation"]))))

        trainer = Seq2SeqTrainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=train_data, 
            eval_dataset=eval_data
        )
        
        print("Starting training...")
        trainer.train()

        # Save the trained model and its tokenizer into artifacts 
        os.makedirs(self.output_model_path, exist_ok=True)
        model.save_pretrained(self.output_model_path)
        tokenizer.save_pretrained(self.output_model_path)
        print(f"Model saved to {self.output_model_path}")
