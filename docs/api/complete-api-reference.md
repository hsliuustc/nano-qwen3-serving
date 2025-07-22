# Complete API Reference

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Types](#data-types)
3. [HTTP API](#http-api)
4. [Error Handling](#error-handling)
5. [Configuration](#configuration)

## Core Classes

### LLM

Main synchronous interface for text generation.

#### Constructor

```python
LLM(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "mps",
    dtype: str = "float16",
    max_queue_size: int = 1000,
    num_blocks: int = 1024,
    block_size: int = 16,
    max_seq_length: int = 4096
)
```

#### Methods

##### generate(prompts, sampling_params=None, priority=NORMAL)

**Parameters:**
- `prompts`: str | List[str] - Input prompts
- `sampling_params`: SamplingParams - Generation parameters
- `priority`: RequestPriority - Request priority

**Returns:** List[Dict[str, Any]] - Generation results

**Example:**
```python
results = llm.generate(
    ["Hello", "How are you?"],
    SamplingParams(temperature=0.7, max_tokens=50)
)
```

##### generate_stream(prompt, sampling_params=None, priority=NORMAL)

**Parameters:**
- `prompt`: str - Input prompt
- `sampling_params`: SamplingParams - Generation parameters
- `priority`: RequestPriority - Request priority

**Yields:** Dict[str, Any] - Streaming tokens

**Example:**
```python
for chunk in llm.generate_stream("Tell a story"):
    print(chunk["token"], end="")
```

##### chat(messages, sampling_params=None, system_prompt="You are a helpful AI assistant.")

**Parameters:**
- `messages`: List[Dict[str, str]] - Conversation messages
- `sampling_params`: SamplingParams - Generation parameters
- `system_prompt`: str - System prompt

**Returns:** Dict[str, Any] - Chat response

**Example:**
```python
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]
response = llm.chat(messages)
```

##### Convenience Methods

```python
generate_single(prompt, sampling_params=None, priority=NORMAL) -> Dict[str, Any]
generate_greedy(prompts, max_tokens=512) -> List[Dict[str, Any]]
generate_creative(prompts, max_tokens=512) -> List[Dict[str, Any]]
generate_balanced(prompts, max_tokens=512) -> List[Dict[str, Any]]
chat_stream(messages, sampling_params=None, system_prompt="...") -> Generator
```

##### Utility Methods

```python
get_stats() -> Dict[str, Any]              # Get performance statistics
get_model_info() -> Dict[str, Any]         # Get model information
clear_stats() -> None                      # Clear statistics
shutdown() -> None                         # Shutdown and cleanup
```

### AsyncLLM

Asynchronous interface for concurrent text generation.

#### Constructor

```python
AsyncLLM(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "mps",
    dtype: str = "float16",
    max_queue_size: int = 1000,
    num_blocks: int = 1024,
    block_size: int = 16,
    max_seq_length: int = 4096,
    worker_count: int = 2
)
```

#### Async Methods

##### async generate(prompts, sampling_params=None, priority=NORMAL)

**Example:**
```python
async with AsyncLLM() as llm:
    result = await llm.generate("Hello, async world!")
```

##### async generate_stream(prompt, sampling_params=None, priority=NORMAL)

**Example:**
```python
async for chunk in llm.generate_stream("Tell a story"):
    print(chunk["token"], end="")
```

##### async generate_batch(prompts, sampling_params=None, batch_size=4, priority=NORMAL)

**Example:**
```python
results = await llm.generate_batch(
    ["Q1", "Q2", "Q3"],
    batch_size=2
)
```

##### Request Management

```python
async submit_request(prompt, sampling_params=None, priority=NORMAL) -> int
async get_result(request_id: int) -> Optional[Dict[str, Any]]
```

### SamplingParams

Configuration for text generation behavior.

#### Constructor

```python
SamplingParams(
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: int = 100,
    min_tokens: int = 0,
    stop_sequences: Optional[List[str]] = None,
    stop_token_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    do_sample: bool = True,
    use_beam_search: bool = False,
    num_beams: int = 1,
    use_cache: bool = True,
    cache_precision: str = "float16"
)
```

#### Class Methods

```python
SamplingParams.greedy(max_tokens=100) -> SamplingParams
SamplingParams.creative(max_tokens=100) -> SamplingParams
SamplingParams.balanced(max_tokens=100) -> SamplingParams
```

#### Methods

```python
to_dict() -> dict                          # Convert to dictionary
__str__() -> str                          # String representation
```

## Data Types

### Generation Result

```python
{
    "request_id": int,           # Unique request identifier
    "prompt": str,               # Original prompt
    "generated_text": str,       # Generated text
    "tokens_generated": int,     # Number of tokens generated
    "total_tokens": int,         # Total tokens (prompt + generated)
    "block_indices": List[int]   # Memory block indices used
}
```

### Streaming Result

```python
{
    "token": str,              # Current token text
    "token_id": int,           # Token ID
    "text": str,               # Accumulated text so far
    "finished": bool,          # Whether generation is complete
    "tokens_generated": int,   # Number of tokens generated so far
    "request_id": int          # Request ID
}
```

### Statistics

```python
{
    "requests_served": int,
    "total_tokens_generated": int,
    "average_response_time": float,
    "cache_hit_rate": float,
    "memory_usage": {
        "allocated": int,
        "cached": int
    },
    "queue_length": int,
    "active_requests": int
}
```

### Model Information

```python
{
    "model_name": str,
    "device": str,
    "dtype": str,
    "max_seq_length": int,
    "vocab_size": int,
    "num_parameters": int
}
```

### RequestPriority (Enum)

```python
RequestPriority.LOW     # Lower priority
RequestPriority.NORMAL  # Default priority
RequestPriority.HIGH    # Higher priority
```

## HTTP API

### Base URL
```
http://127.0.0.1:8000
```

### Chat Completions

#### POST /v1/chat/completions

**Request:**
```json
{
  "model": "qwen3-0.6b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 100,
  "stream": false,
  "stop": null
}
```

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
        "content": "Hello! How can I help you?"
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

### Completions (Legacy)

#### POST /v1/completions

**Request:**
```json
{
  "model": "qwen3-0.6b",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Models

#### GET /v1/models

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

### Health and Stats

#### GET /health

**Response:**
```json
{
  "status": "healthy",
  "uptime": 3600.5,
  "model_info": {...},
  "memory_usage": {...}
}
```

#### GET /stats

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

## Error Handling

### Common Exceptions

```python
# Import exceptions
from nano_qwen3_serving.exceptions import (
    ModelLoadError,
    GenerationError,
    QueueFullError,
    InvalidParametersError
)

try:
    result = llm.generate("Hello")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except GenerationError as e:
    print(f"Generation failed: {e}")
except QueueFullError as e:
    print(f"Queue is full: {e}")
except InvalidParametersError as e:
    print(f"Invalid parameters: {e}")
```

### HTTP Error Responses

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Missing required parameter: messages",
    "code": "missing_parameter"
  }
}
```

**Common HTTP Status Codes:**
- `400`: Bad Request - Invalid parameters
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server error
- `503`: Service Unavailable - Service not ready

## Configuration

### Environment Variables

```bash
# Model configuration
MODEL_NAME=Qwen/Qwen3-0.6B
DEVICE=mps
DTYPE=float16

# Memory configuration
MAX_QUEUE_SIZE=1000
NUM_BLOCKS=1024
BLOCK_SIZE=16
MAX_SEQ_LENGTH=4096

# Server configuration
HOST=127.0.0.1
PORT=8000
WORKERS=1
LOG_LEVEL=info
```

### Command Line Options

```bash
python tools/start_service.py \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 4 \
    --reload \
    --log-level debug \
    --model-name Qwen/Qwen3-0.6B \
    --device mps \
    --dtype float16
```

### Configuration File

```yaml
# config.yaml
model:
  name: "Qwen/Qwen3-0.6B"
  device: "mps"
  dtype: "float16"

memory:
  max_queue_size: 1000
  num_blocks: 1024
  block_size: 16
  max_seq_length: 4096

server:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  log_level: "info"
```

## Complete Example

```python
import asyncio
from nano_qwen3_serving import LLM, AsyncLLM, SamplingParams
from nano_qwen3_serving.core.scheduler import RequestPriority

def sync_example():
    """Synchronous example."""
    with LLM(model_name="Qwen/Qwen3-0.6B") as llm:
        # Single generation
        result = llm.generate_single(
            "Explain machine learning",
            SamplingParams.balanced(max_tokens=200)
        )
        print(f"Response: {result['generated_text']}")
        
        # Batch generation
        prompts = ["What is AI?", "How do neural networks work?"]
        results = llm.generate(prompts, SamplingParams.greedy(max_tokens=100))
        
        # Streaming
        for chunk in llm.generate_stream("Tell a story"):
            print(chunk["token"], end="")
            if chunk["finished"]:
                break
        
        # Chat
        messages = [{"role": "user", "content": "Hello!"}]
        chat_result = llm.chat(messages)
        
        # Statistics
        stats = llm.get_stats()
        print(f"Generated {stats['total_tokens_generated']} tokens")

async def async_example():
    """Asynchronous example."""
    async with AsyncLLM() as llm:
        # Concurrent requests
        tasks = [
            llm.generate("Question 1"),
            llm.generate("Question 2"),
            llm.generate("Question 3")
        ]
        results = await asyncio.gather(*tasks)
        
        # Batch processing
        batch_results = await llm.generate_batch(
            ["Q1", "Q2", "Q3", "Q4"],
            batch_size=2
        )
        
        # Async streaming
        async for chunk in llm.generate_stream("Write a poem"):
            print(chunk["token"], end="")

# Run examples
sync_example()
asyncio.run(async_example())
```

This complete API reference covers all public interfaces, data types, and usage patterns for the nano-qwen3-serving library.
