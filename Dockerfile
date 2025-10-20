FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD MODEL DURING BUILD (this is the key change)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('TheBloke/Llama-3-8B-Instruct-GPTQ'); \
    AutoModelForCausalLM.from_pretrained('TheBloke/Llama-3-8B-Instruct-GPTQ')"

# Copy server code
COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]