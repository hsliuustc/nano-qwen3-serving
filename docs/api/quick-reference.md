# Quick Reference Guide

## Import Statements

```python
from nano_qwen3_serving import LLM, AsyncLLM, SamplingParams
from nano_qwen3_serving.core.scheduler import RequestPriority
```

## Basic Usage

### Synchronous

```python
# Initialize
llm = LLM()

# Generate text
result = llm.generate_single("Hello world")
print(result["generated_text"])

# Streaming
for chunk in llm.generate_stream("Tell a story"):
    print(chunk["token"], end="")

# Chat
messages = [{"role": "user", "content": "Hi!"}]
response = llm.chat(messages)

# Cleanup
llm.shutdown()
```

### Context Manager

```python
with LLM() as llm:
    result = llm.generate_single("Hello")
    print(result["generated_text"])
```

### Asynchronous

```python
async with AsyncLLM() as llm:
    result = await llm.generate("Hello async world")
    print(result["generated_text"])
```

## Sampling Parameters

```python
# Presets
SamplingParams.greedy(max_tokens=100)     # Deterministic
SamplingParams.creative(max_tokens=200)   # High creativity
SamplingParams.balanced(max_tokens=150)   # Balanced

# Custom
SamplingParams(
    temperature=0.7,      # Randomness (0.0-2.0)
    top_p=0.9,           # Nucleus sampling (0.0-1.0)
    max_tokens=200,      # Max tokens to generate
    stop_sequences=[".", "!", "?"]  # Stop sequences
)
```

## HTTP API

### Start Server

```bash
python tools/start_service.py --port 8000
```

### Chat Completions

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### Python Requests

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
print(response.json()["choices"][0]["message"]["content"])
```

### OpenAI Client

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Common Patterns

### Task-Specific Configurations

```python
# Factual Q&A
factual = SamplingParams(temperature=0.3, max_tokens=150)

# Creative writing
creative = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=500)

# Code generation
code = SamplingParams(temperature=0.1, stop_sequences=["```"])
```

### Error Handling

```python
try:
    with LLM() as llm:
        result = llm.generate_single("Hello")
except Exception as e:
    print(f"Error: {e}")
```

### Performance Monitoring

```python
with LLM() as llm:
    # Generate text
    result = llm.generate_single("Hello")
    
    # Check stats
    stats = llm.get_stats()
    print(f"Tokens generated: {stats['total_tokens_generated']}")
    print(f"Avg response time: {stats['average_response_time']:.2f}s")
```

## Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/stats` | GET | Performance stats |

## Configuration

### Environment Variables

```bash
export MODEL_NAME="Qwen/Qwen3-0.6B"
export DEVICE="mps"
export PORT=8000
```

### Command Line

```bash
python tools/start_service.py \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 4
```

## Troubleshooting

### Common Issues

1. **Service not starting**: Check model path and device availability
2. **Out of memory**: Reduce `num_blocks` or `max_seq_length`
3. **Slow generation**: Check device (use "mps" on Apple Silicon)
4. **Connection refused**: Verify service is running on correct port

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Debug Mode

```bash
python tools/start_service.py --log-level debug
```
