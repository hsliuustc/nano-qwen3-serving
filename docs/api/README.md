# Comprehensive API Documentation

Welcome to the comprehensive API documentation for Nano Qwen3 Serving. This documentation covers all public APIs, functions, and components with detailed examples and usage instructions.

## 📚 Documentation Structure

### [Core Components](core-components.md)
- **LLM Class**: Main synchronous interface for text generation
- **SamplingParams**: Configuration for generation behavior
- **Utility methods and context management**

### [Async Components](async-components.md)  
- **AsyncLLM Class**: Asynchronous interface for concurrent processing
- **Async streaming and batch processing**
- **Request management and async context managers**

### [HTTP API](http-api.md)
- **OpenAI-compatible endpoints** (`/v1/chat/completions`, `/v1/completions`)
- **Health monitoring** (`/health`, `/stats`, `/v1/models`)
- **Streaming responses and error handling**

### [Usage Examples](usage-examples.md)
- **Basic usage patterns** for common tasks
- **Advanced configurations** for specific use cases
- **Error handling and performance monitoring**

### [Service Configuration](service-configuration.md)
- **Command-line options** and environment variables
- **Production deployment** best practices
- **Load testing and monitoring**

## 🚀 Quick Start

### Installation

```bash
pip install nano-qwen3-serving
```

### Basic Usage

```python
from nano_qwen3_serving import LLM, SamplingParams

# Initialize LLM
with LLM() as llm:
    # Generate text
    result = llm.generate_single(
        "Explain machine learning",
        sampling_params=SamplingParams.balanced(max_tokens=200)
    )
    print(result["generated_text"])
```

### Start HTTP Service

```bash
python tools/start_service.py --port 8000
```

### Use HTTP API

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

## 🎯 Key Features

### Synchronous API
- **Simple interface** for basic text generation
- **Streaming support** for real-time output
- **Chat functionality** with conversation history
- **Preset configurations** for common use cases

### Asynchronous API
- **High concurrency** for multiple requests
- **Batch processing** for efficient throughput
- **Async streaming** for real-time applications
- **Request management** with submit/retrieve pattern

### HTTP API
- **OpenAI compatibility** for drop-in replacement
- **Standard endpoints** following OpenAI specification
- **Health monitoring** and performance metrics
- **Streaming responses** with Server-Sent Events

### Performance Features
- **Apple Silicon optimization** with MPS acceleration
- **Memory management** with configurable KV cache
- **Request scheduling** with priority queues
- **Performance monitoring** with detailed statistics

## 📋 API Reference Summary

### Main Classes

| Class | Purpose | Usage |
|-------|---------|-------|
| `LLM` | Synchronous text generation | `llm.generate("Hello")` |
| `AsyncLLM` | Asynchronous text generation | `await llm.generate("Hello")` |
| `SamplingParams` | Generation configuration | `SamplingParams(temperature=0.7)` |

### HTTP Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/stats` | GET | Performance stats |

### Key Methods

| Method | Class | Purpose |
|--------|-------|---------|
| `generate()` | LLM/AsyncLLM | Generate text |
| `generate_stream()` | LLM/AsyncLLM | Streaming generation |
| `chat()` | LLM/AsyncLLM | Chat responses |
| `generate_batch()` | AsyncLLM | Batch processing |

## 💡 Usage Patterns

### Task-Specific Configurations

```python
# Factual Q&A
factual_params = SamplingParams(temperature=0.3, max_tokens=150)

# Creative writing
creative_params = SamplingParams.creative(max_tokens=500)

# Code generation
code_params = SamplingParams(temperature=0.1, stop_sequences=["```"])

# Technical documentation
tech_params = SamplingParams(temperature=0.2, repetition_penalty=1.1)
```

### Production Deployment

```bash
# Production server
python tools/start_service.py \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level warning

# With environment variables
export MODEL_NAME="Qwen/Qwen3-0.6B"
export DEVICE="mps"
export MAX_QUEUE_SIZE=2000
python tools/start_service.py
```

### Error Handling

```python
try:
    with LLM() as llm:
        result = llm.generate_single("Hello")
        print(result["generated_text"])
except Exception as e:
    print(f"Generation failed: {e}")
```

## 🔧 Advanced Features

### Custom Sampling Strategies
- Temperature control for creativity vs coherence
- Top-p and top-k sampling for token selection
- Stop sequences for controlled generation
- Repetition penalties for diverse output

### Memory Management
- Configurable KV cache size
- Block-based memory allocation
- Automatic cleanup and garbage collection
- Memory usage monitoring

### Request Scheduling
- Priority-based request queuing
- Concurrent request handling
- Batch processing optimization
- Load balancing across workers

### Performance Monitoring
- Real-time statistics collection
- Token generation metrics
- Response time tracking
- Memory usage monitoring

## 📖 Additional Resources

- [Project README](../../README.md) - Project overview and setup
- [Examples](../../examples/) - Complete working examples
- [Getting Started Guide](../getting-started/) - Step-by-step tutorials
- [Troubleshooting](../troubleshooting/) - Common issues and solutions

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to the documentation and codebase.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.
