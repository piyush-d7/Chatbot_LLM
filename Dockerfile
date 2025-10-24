FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD with vLLM (not transformers)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', \
    local_files_only=False, \
    resume_download=True)"

# Copy server code
COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
