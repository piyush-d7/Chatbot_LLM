# from fastapi import FastAPI
# from pydantic import BaseModel
# from vllm import LLM, SamplingParams
# import time

# app = FastAPI(title="LLM Inference Service")

# # Load model with vLLM
# # MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
# # MODEL_NAME = "TheBloke/Llama-3-8B-Instruct-GPTQ"
# MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
# print("Loading model with vLLM...")
# llm = LLM(
#     model=MODEL_NAME,
#     trust_remote_code=True,
#     gpu_memory_utilization=0.90,  # Reduced slightly for safety
#     # quantization="gptq",  # CRITICAL: Tell vLLM this is a GPTQ model
#     dtype="float16",
#     max_model_len=4096
# )
# print("Model loaded successfully")

# class GenerateRequest(BaseModel):
#     prompt: str
#     max_tokens: int = 512
#     temperature: float = 0.7

# class GenerateResponse(BaseModel):
#     generated_text: str
#     processing_time: float

# @app.get("/health")
# def health_check():
#     return {"status": "healthy", "model": MODEL_NAME}

# @app.post("/generate", response_model=GenerateResponse)
# def generate(request: GenerateRequest):
#     start_time = time.time()
    
#     sampling_params = SamplingParams(
#         temperature=request.temperature,
#         max_tokens=request.max_tokens,
#         top_p=0.95,
#         repetition_penalty=1.1
#     )
    
#     outputs = llm.generate([request.prompt], sampling_params)
#     generated_text = outputs[0].outputs[0].text
    
#     processing_time = time.time() - start_time
    
#     return GenerateResponse(
#         generated_text=generated_text,
#         processing_time=processing_time
#     )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import time
from typing import List, Dict

app = FastAPI(title="LLM Inference Service")

# Load model with vLLM
# MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

print("Loading model with vLLM...")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    dtype="float16",
    quantization="gptq",
    max_model_len=4096
)
print("Model loaded successfully")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatMessage(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    generated_text: str
    processing_time: float

class ChatResponse(BaseModel):
    response: str
    processing_time: float

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """
    Format messages into Qwen chat template format
    Qwen uses: <|im_start|>role\ncontent<|im_end|>
    """
    formatted_prompt = ""
    
    for message in messages:
        role = message.role
        content = message.content
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add assistant start token to prompt completion
    formatted_prompt += "<|im_start|>assistant\n"
    
    return formatted_prompt

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Legacy endpoint for raw prompt generation"""
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

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat endpoint that accepts messages in OpenAI format
    """
    start_time = time.time()
    
    # Format messages into Qwen chat template
    formatted_prompt = format_chat_prompt(request.messages)
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=0.95,
        repetition_penalty=1.05,
        stop=["<|im_end|>", "<|endoftext|>"]  # Stop tokens for Qwen
    )
    
    outputs = llm.generate([formatted_prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    
    processing_time = time.time() - start_time
    
    return ChatResponse(
        response=generated_text,
        processing_time=processing_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

**Key changes:**

1. **Added `ChatMessage` and `ChatRequest` models** - Accept messages in OpenAI format
2. **Added `format_chat_prompt()` function** - Converts messages to Qwen's chat template format:
```
   <|im_start|>system
   You are a helpful assistant<|im_end|>
   <|im_start|>user
   Hello!<|im_end|>
   <|im_start|>assistant
'''