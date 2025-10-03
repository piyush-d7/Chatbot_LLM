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
    input_length = inputs.input_ids.shape[1]
    
    print(f"Input length: {input_length}")
    print(f"Input tokens: {inputs.input_ids}")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output tokens: {outputs[0]}")
    
    # Extract only generated tokens
    generated_ids = outputs[0][input_length:]
    print(f"Generated tokens only: {generated_ids}")
    print(f"Generated tokens length: {len(generated_ids)}")
    
    # Decode
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Decoded text: '{response_text}'")
    
    processing_time = time.time() - start_time
    
    return GenerateResponse(
        generated_text=response_text,
        processing_time=processing_time
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)