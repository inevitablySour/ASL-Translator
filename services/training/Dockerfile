FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY retrain_worker.py .

# Copy source code from src directory (model trainer, data loader, etc.)
COPY src/ ./src/

CMD ["python", "-u", "retrain_worker.py", "--interval", "300", "--threshold", "200"]
