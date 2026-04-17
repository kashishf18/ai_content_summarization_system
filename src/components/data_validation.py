import os
import json
from datasets import load_from_disk

class DataValidation:
    def __init__(self):
        self.data_path = "artifacts/data/raw"
        self.report_path = "artifacts/data/validation_report.json"
        self.required_splits = ["train", "test", "validation"]
        self.required_columns = ["id", "dialogue", "summary"]

    def validate(self):
        print("Starting data validation...")
        report = {"status": "success", "errors": []}
        
        try:
            dataset = load_from_disk(self.data_path)
            
            # Check splits
            for split in self.required_splits:
                if split not in dataset:
                    report["errors"].append(f"Missing split: {split}")
                else:
                    # Check columns and minimum sample counts
                    ds_split = dataset[split]
                    if len(ds_split) < 50:  # arbitrary positive min count
                        report["errors"].append(f"Split {split} has less than 50 samples.")
                    
                    for col in self.required_columns:
                        if col not in ds_split.column_names:
                            report["errors"].append(f"Missing column {col} in split {split}")
            
            if report["errors"]:
                report["status"] = "failed"
                
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=4)
                
            print(f"Validation completed. Status: {report['status']}")
            return report['status'] == 'success'
            
        except Exception as e:
            report["status"] = "error"
            report["errors"].append(str(e))
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=4)
            print(f"Validation encountered an error: {e}")
            return False
