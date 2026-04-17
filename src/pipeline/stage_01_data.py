import sys
import os

# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

class DataPipeline:
    def __init__(self):
        pass

    def run(self):
        print("====== Stage 01: Data Pipeline Started ======")
        
        ingestion = DataIngestion()
        ingestion.download_data()
        
        validation = DataValidation()
        is_valid = validation.validate()
        
        if not is_valid:
            print("Data Validation failed. Stopping pipeline. Check artifacts/data/validation_report.json.")
            return

        transformation = DataTransformation()
        transformation.process()
        
        print("====== Stage 01: Data Pipeline Completed ======")

if __name__ == '__main__':
    pipeline = DataPipeline()
    pipeline.run()
