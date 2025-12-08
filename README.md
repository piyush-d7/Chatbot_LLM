# Chatbot LLM Inference Service

This microservice handles the **generation** phase of the RAG pipeline. It hosts a quantized Large Language Model (LLM) on a GPU-accelerated container to generate natural, context-aware responses to user queries.

## 🚀 Features

  * **High-Performance Inference**: Built on **vLLM**, a high-throughput and memory-efficient inference engine that offers 10-20x speed improvements over standard HuggingFace transformers.
  * **Optimized Model**: Uses **Qwen 2.5-7B Instruct (GPTQ Int4)**. The 4-bit quantization significantly reduces memory usage (fitting on a 16GB GPU) while maintaining high accuracy.
  * **OpenAI Compatible**: Exposes a `/chat` endpoint compatible with the OpenAI Chat Completion API format, making integration seamless.
  * **Serverless GPU**: Designed for **Alibaba Cloud Function Compute (GPU)**, allowing for scalable, high-performance AI without managing persistent servers.

## 🛠️ Tech Stack

  * **Engine**: [vLLM](https://github.com/vllm-project/vllm)
  * **Model**: `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` (\~11.9GB)
  * **Quantization**: GPTQ (4-bit)
  * **Container**: Docker
  * **Hardware Req**: NVIDIA GPU (Tesla T4 or better)

-----

## ⚙️ Configuration & Hardware

This service is tuned for specific hardware constraints to balance performance and cost.

| Component | Specification |
| :--- | :--- |
| **Platform** | Alibaba Cloud Function Compute (FC) |
| **Docker Image** | `piyushwebsitetoon/chatbot-llm:latest` |
| **GPU** | NVIDIA Tesla T4 (16GB VRAM) |
| **vCPU / RAM** | 4 vCPU / 16GB RAM |
| **Disk** | 10GB (Ephemeral) |
| **Concurrency** | 5-10 concurrent requests |
| **Cold Start** | **None** (Min Instances: 1 provisioned) |

-----

## 🏃 Deployment

### 1\. Pull the Image

The image is pre-built with vLLM and the model weights baked in to ensure fast startup.

```bash
docker pull piyushwebsitetoon/chatbot-llm:latest
```

### 2\. Run Locally (Requires GPU)

To test this service locally, you must have a machine with an NVIDIA GPU and the NVIDIA Container Toolkit installed.

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  piyushwebsitetoon/chatbot-llm:latest \
  --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --quantization gptq \
  --dtype float16
```

### 3\. Deploy to Alibaba Cloud FC

1.  **Select Runtime**: Custom Container.
2.  **Image**: `piyushwebsitetoon/chatbot-llm:latest`.
3.  **Instance Type**: **GPU Accelerated** (fc.gpu.tesla.1).
4.  **Resources**: 16GB Memory, 4 vCPU.
5.  **Port**: 8000.
6.  **Idle Provisioning**: Set **Min Instances = 1** to keep the model loaded in VRAM (critical for avoiding 2-3 minute cold starts).
7.  **Timeout**: Set to `360` seconds.

-----

## 🔌 API Endpoints

### Chat Completion (OpenAI Format)

**POST** `/chat`

The primary endpoint for generating answers.

**Request:**

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
  "messages": [
    {"role": "system", "content": "You are a helpful HVAC assistant."},
    {"role": "user", "content": "Why is my furnace making a banging noise?"}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Response:**

```json
{
  "id": "chatcmpl-123...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A banging noise in your furnace is often caused by delayed ignition..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Raw Generation (Legacy)

**POST** `/generate`

**Request:**

```json
{
  "prompt": "The capital of France is",
  "max_tokens": 10
}
```

### Health Check

**GET** `/health`

Returns HTTP 200 if the vLLM engine is running and ready to accept requests.

-----

## 📊 Performance Metrics

  * **Throughput**: 50-100 tokens/second (depending on prompt size).
  * **Latency**: 2-5 seconds Time to First Token (TTFT).
  * **Speedup**: Approx. 10-20x faster than standard HuggingFace Transformers inference due to vLLM's PagedAttention.

## 📂 Project Structure

```text
.
├── Dockerfile           # Sets up vLLM environment & downloads model
├── requirements.txt     # Python dependencies (vllm, etc.)
└── scripts/             # Startup scripts for the container
```
