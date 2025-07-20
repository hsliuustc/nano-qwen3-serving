# API Reference

Nano Qwen3 Serving provides a fully OpenAI-compatible API. All endpoints follow the OpenAI API specification, making it a drop-in replacement for OpenAI services.

## 🔗 Base URL

```
http://localhost:8000
```

## 📋 Authentication

Currently, Nano Qwen3 Serving doesn't require authentication for local development. For production deployments, consider implementing API key authentication.

## 📊 Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and server status |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Generate chat completions |
| `/v1/chat/completions` | POST | Generate streaming completions |
| `/stats` | GET | Performance statistics |

## 🏥 Health Check

### GET `/health`

Check if the server is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "model": "Qwen/Qwen3-0.6B",
  "device": "mps",
  "uptime": 3600
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

## 🤖 Models

### GET `/v1/models`

List all available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-0.6B",
      "object": "model",
      "created": 1705312800,
      "owned_by": "nano-qwen3-serving",
      "permission": [],
      "root": "Qwen/Qwen3-0.6B",
      "parent": null
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/v1/models
```

## 💬 Chat Completions

### POST `/v1/chat/completions`

Generate chat completions with the specified model.

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g., "Qwen/Qwen3-0.6B") |
| `messages` | array | Yes | Array of message objects |
| `stream` | boolean | No | Whether to stream the response (default: false) |
| `max_tokens` | integer | No | Maximum tokens to generate (default: 2048) |
| `temperature` | number | No | Sampling temperature (0.0-2.0, default: 1.0) |
| `top_p` | number | No | Nucleus sampling parameter (0.0-1.0, default: 1.0) |
| `n` | integer | No | Number of completions to generate (default: 1) |
| `stop` | string/array | No | Stop sequences |
| `presence_penalty` | number | No | Presence penalty (-2.0 to 2.0, default: 0.0) |
| `frequency_penalty` | number | No | Frequency penalty (-2.0 to 2.0, default: 0.0) |
| `logit_bias` | object | No | Logit bias for specific tokens |
| `user` | string | No | User identifier for tracking |

#### Message Object

```json
{
  "role": "system|user|assistant",
  "content": "message content"
}
```

#### Non-Streaming Response

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1705312800,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm here to help you."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

#### Streaming Response

When `stream=true`, the response is a Server-Sent Events (SSE) stream:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705312800,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705312800,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705312800,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"content":" I'm"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705312800,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Examples

#### Basic Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

#### Streaming Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Write a short poem about AI."}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

#### Python Example

```python
import requests

# Non-streaming
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])

# Streaming
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "user", "content": "Write a story about a robot."}
        ],
        "stream": True,
        "max_tokens": 300
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]  # Remove 'data: ' prefix
            if data == '[DONE]':
                break
            try:
                chunk = json.loads(data)
                if 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
            except json.JSONDecodeError:
                continue
```

## 📊 Performance Statistics

### GET `/stats`

Get real-time performance statistics.

**Response:**
```json
{
  "requests_processed": 150,
  "tokens_generated": 2500,
  "average_response_time": 0.045,
  "requests_per_second": 22.5,
  "memory_usage_mb": 2048,
  "gpu_utilization": 0.75,
  "model_info": {
    "name": "Qwen/Qwen3-0.6B",
    "parameters": 596049920,
    "device": "mps"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/stats
```

## ⚠️ Error Responses

All endpoints return standard HTTP status codes and error messages in the following format:

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `model_not_found` | Specified model doesn't exist |
| `invalid_request_error` | Request format is invalid |
| `rate_limit_exceeded` | Too many requests |
| `server_error` | Internal server error |

## 🔧 Rate Limiting

Currently, Nano Qwen3 Serving doesn't implement rate limiting. For production use, consider implementing rate limiting based on your requirements.

## 📝 Request/Response Headers

### Request Headers

| Header | Description |
|--------|-------------|
| `Content-Type` | Must be `application/json` |
| `Accept` | `application/json` for non-streaming, `text/event-stream` for streaming |

### Response Headers

| Header | Description |
|--------|-------------|
| `Content-Type` | `application/json` or `text/event-stream` |
| `Cache-Control` | Caching directives |
| `X-Request-ID` | Unique request identifier |

## 🔄 WebSocket Support

WebSocket support is planned for future releases. Currently, use Server-Sent Events (SSE) for streaming responses.

## 📚 Next Steps

- **[Streaming Guide](streaming.md)**: Detailed streaming implementation
- **[Models Guide](models.md)**: Model-specific configurations
- **[Examples](../examples/basic-examples.md)**: More usage examples
- **[Troubleshooting](../troubleshooting/common-issues.md)**: Common issues and solutions 