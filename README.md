# AI Content Summarization System

An end-to-end automated NLP pipeline for abstractive text summarization using the T5-small model. This project features a modular architecture, FastAPI for serving, and is production-ready for deployment on AWS.

## 🚀 Features

- **T5-small model**: abstractive summarization.
- **FastAPI**: High-performance API for health checks, training, and prediction.
- **Lazy Loading**: Efficient model loading using FastAPI lifespan.
- **Dockerized**: Easy setup and deployment using Docker containers.
- **Cloud Ready**: Configured for Amazon ECR and EC2 (ap-south-1).
- **Modular Pipeline**: Clean separation between data ingestion, validation, training, and prediction.

---

## 🛠️ Technology Stack

- **NLP**: Hugging Face Transformers (T5)
- **Framework**: FastAPI (Pydantic, Uvicorn)
- **Ops**: Docker, Amazon ECR/EC2
- **Config**: YAML based configuration (`config.yaml`)

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.8+
- Docker (optional, for containerization)
- AWS CLI configured (for cloud deployment)

### 2. local Environment
```bash
# Clone the repository
git clone https://github.com/kashishf18/ai_content_summarization_system.git
cd ai_content_summarization_system

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory (refer to `.env.example`):
```env
HF_TOKEN=your_token
AWS_ACCESS_KEY_ID=your_id
AWS_SECRET_ACCESS_KEY=your_key
AWS_DEFAULT_REGION=ap-south-1
# ... other vars
```

---

## 📦 Docker Usage

### Build the Image
```bash
docker build -t summarizer .
```

### Run Locally
```bash
docker run -d --name summarizer_app --env-file .env -p 8080:8080 summarizer
```

---

## 📡 API Endpoints

- **GET `/health`**: Check system status. Returns `ready` if the model is loaded.
- **POST `/predict`**: Generate a summary. 
  - *Payload*: `{"text": "Your long text here"}`
- **GET `/train`**: Initiate background fine-tuning.
- **GET `/docs`**: Interactive Swagger UI.

---

## ☁️ AWS Deployment

The project is designed to run on a **t3.micro** EC2 instance with an attached IAM Role for ECR access.

1. **Push to ECR**: Tag and push your image to your Amazon ECR repository.
2. **Launch EC2**: Use an Amazon Linux 2 AMI with a **20GB** root volume.
3. **Run**: Pull and run the container on the EC2 instance.

---

## 📄 License
This project is licensed under the MIT License.
