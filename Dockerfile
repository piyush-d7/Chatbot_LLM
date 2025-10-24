# FROM python:3.11-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # PRE-DOWNLOAD with vLLM (not transformers)
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', \
#     local_files_only=False, \
#     resume_download=True)"

# # Copy server code
# COPY server.py .

# EXPOSE 8000

# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

# Minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Optimized requirements installation
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    vllm \
    pydantic \
    && pip cache purge \
    && rm -rf /root/.cache/pip \
    && find /usr/local/lib/python3.11 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Download model with minimal files
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download( \
        'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', \
        cache_dir='/root/.cache/huggingface', \
        ignore_patterns=['*.md', '*.txt', '*.pdf', 'LICENSE*', 'README*', '*.gitattributes'] \
    )" \
    && find /root/.cache/huggingface -name "*.md" -delete \
    && find /root/.cache/huggingface -name "*.txt" -delete \
    && find /root/.cache/huggingface -name "*.pdf" -delete

# Copy server
COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]