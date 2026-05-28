@echo off
:: Run the FastAPI application using the virtual environment's python
echo Starting AI Content Summarization System...
echo Web UI will open automatically at: http://127.0.0.1:8080
start http://127.0.0.1:8080
venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
