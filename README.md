# Wakil AI - Local Neural Engine

A fully local AI engine powered by **Microsoft Phi-3-mini**. No API keys, no internet, 100% private.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python server.py
```

The server will:
- Download the Phi-3-mini model on **first run** (~2.3 GB)
- Start on **http://localhost:8080**
- Show API docs at **http://localhost:8080/docs**

## API Usage

The server is fully **OpenAI-compatible**. Send requests to:

```
POST http://localhost:8080/v1/chat/completions
```

### Example request:
```json
{
  "model": "wakil-local-1",
  "messages": [
    {"role": "user", "content": "Hello, who are you?"}
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

## System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM       | 8 GB    | 16 GB       |
| GPU VRAM  | -       | 4 GB (CUDA) |
| Storage   | 5 GB    | 10 GB       |
| Python    | 3.9+    | 3.11+       |

## Endpoints
- `GET /` — Engine status
- `GET /health` — Health check
- `POST /v1/chat/completions` — Chat API
- `GET /docs` — Interactive API docs (Swagger)
