# HTTP API Documentation

## Base Configuration

**Default URL:** `http://127.0.0.1:8000`

**Common Headers:**
```http
Content-Type: application/json
Accept: application/json
```

## Chat Completions

### POST /v1/chat/completions

Generate chat completions.

**Request Body:**
```json
{
  "model": "qwen3-0.6b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

**Parameters:**
- `model`: Model name (required)
- `messages`: Array of message objects (required)
- `temperature`: Sampling temperature (0.0-2.0)
- `top_p`: Top-p sampling (0.0-1.0)
- `max_tokens`: Maximum tokens to generate
- `stream`: Enable streaming (boolean)
- `stop`: Stop sequences (string or array)

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

### Streaming Response

When `stream: true`, response is sent as Server-Sent Events:

```
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

## Examples

### Python with requests

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }
)
result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Streaming with Python

```python
import requests
import json

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
        "max_tokens": 200
    },
    headers={"Accept": "text/event-stream"},
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
            except json.JSONDecodeError:
                continue
```

### OpenAI Python Client (Drop-in Replacement)

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=50
)

print(response.choices[0].message.content)
```

## Health and Monitoring

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "uptime": 3600.5,
  "model_info": {
    "model_name": "Qwen/Qwen3-0.6B",
    "device": "mps",
    "dtype": "float16"
  }
}
```

### GET /stats

Performance statistics.

**Response:**
```json
{
  "requests_served": 1250,
  "total_tokens_generated": 45000,
  "average_response_time": 0.85,
  "queue_length": 3,
  "active_requests": 2
}
```

### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-0.6b",
      "object": "model",
      "created": 1677652288,
      "owned_by": "nano-qwen3-serving"
    }
  ]
}
```
