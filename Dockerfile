FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install build essential + system dependencies
# These might not be strictly necessary, but good practice for any C-extensions or network calls
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to cache the pip install step
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code (omits files in .dockerignore)
COPY . .

# Expose the API port
EXPOSE 8080

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
