from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI(title="LLM Inference Service")

# Load model on startup
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    generated_text: str
    processing_time: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    import time
    start_time = time.time()
    
    # Tokenize input
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True
        )
    
    # Decode output
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    processing_time = time.time() - start_time
    
    return GenerateResponse(
        generated_text=response_text,
        processing_time=processing_time
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)