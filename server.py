# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import uvicorn

# app = FastAPI(title="LLM Inference Service")

# # Load model on startup
# MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
# print("Loading model...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, 
#     torch_dtype=torch.float16,
# )
# print("Model loaded successfully")

# # Request/Response models
# class GenerateRequest(BaseModel):
#     prompt: str
#     max_tokens: int = 256
#     temperature: float = 0.7

# class GenerateResponse(BaseModel):
#     generated_text: str
#     processing_time: float

# @app.get("/health")
# def health_check():
#     return {"status": "healthy", "model": MODEL_NAME}

# @app.post("/generate", response_model=GenerateResponse)
# def generate(request: GenerateRequest):
#     import time
#     start_time = time.time()
    
#     inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
#     input_length = inputs.input_ids.shape[1]
    
#     print(f"Input length: {input_length}")
#     print(f"Generating...")
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=min(request.max_tokens, 100),  # Limit to 100 for speed
#             temperature=0.7,  # Fixed temperature
#             do_sample=False,  # Greedy decoding = faster
#             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     print(f"Generation complete")
    
#     generated_ids = outputs[0][input_length:]
#     response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
#     processing_time = time.time() - start_time
#     print(f"Processing time: {processing_time:.2f}s")
#     print(f"Generated text: {response_text[:100]}...")
    
#     return GenerateResponse(
#         generated_text=response_text,
#         processing_time=processing_time
#     )

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import time

app = FastAPI(title="LLM Inference Service")

# Load model with vLLM
# MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
print("Loading model with vLLM...")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=0.95,
    dtype="float16"
)
print("Model loaded successfully")

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
    start_time = time.time()
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=0.95
    )
    
    outputs = llm.generate([request.prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    processing_time = time.time() - start_time
    
    return GenerateResponse(
        generated_text=generated_text,
        processing_time=processing_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)