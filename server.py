from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import time

app = FastAPI(title="LLM Inference Service")

# Load model with vLLM
# MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MODEL_NAME = "TheBloke/Llama-3-8B-Instruct-GPTQ"
print("Loading model with vLLM...")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=0.90,  # Reduced slightly for safety
    quantization="gptq",  # CRITICAL: Tell vLLM this is a GPTQ model
    dtype="float16",
    max_model_len=4096
)
print("Model loaded successfully")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
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
        top_p=0.95,
        repetition_penalty=1.1
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