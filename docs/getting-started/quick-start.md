# Quick Start Guide

Get up and running with Nano Qwen3 Serving in under 5 minutes! This guide will walk you through installation, basic setup, and your first API request.

## 🎯 Prerequisites

- **macOS with Apple Silicon** (M1, M2, M3, or M1 Pro/Max/Ultra)
- **Python 3.8+** installed
- **8GB+ RAM** (16GB+ recommended)
- **Internet connection** for model download

## 🚀 Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install nano-qwen3-serving
```

### Option 2: Install from Source

```bash
git clone https://github.com/hsliuustc/nano-qwen3-serving.git
cd nano-qwen3-serving
pip install -e .
```

## ⚡ Quick Start

### 1. Start the Server

```bash
# Start with default settings (port 8000)
python -m nano_qwen3_serving

# Or specify a custom port
python -m nano_qwen3_serving --port 8001

# Start with a specific model
python -m nano_qwen3_serving --model Qwen/Qwen3-1.5B
```

You should see output like:
```
🚀 Starting nano Qwen3 Serving Service
📊 Model: Qwen/Qwen3-0.6B
🔧 Device: mps
🌐 Host: 127.0.0.1
🔌 Port: 8000
👥 Workers: 1
📝 Log Level: info
--------------------------------------------------
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 2. Test the Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

### 3. Make Your First Request

#### Using curl:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Hello! How are you today?"}
    ],
    "max_tokens": 100
  }'
```

#### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "user", "content": "Hello! How are you today?"}
        ],
        "max_tokens": 100
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

## 🔧 Configuration Options

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | `8000` |
| `--host` | Server host | `127.0.0.1` |
| `--model` | Model to load | `Qwen/Qwen3-0.6B` |
| `--device` | Device (mps/cpu) | `mps` |
| `--workers` | Number of workers | `1` |
| `--log-level` | Logging level | `info` |

### Environment Variables

```bash
export NANO_QWEN3_PORT=8001
export NANO_QWEN3_MODEL=Qwen/Qwen3-1.5B
export NANO_QWEN3_DEVICE=mps
export NANO_QWEN3_LOG_LEVEL=debug
```

## 📊 Available Models

| Model | Parameters | Memory | Speed | Use Case |
|-------|------------|--------|-------|----------|
| `Qwen/Qwen3-0.6B` | 596M | ~2GB | Fast | Development, Testing |
| `Qwen/Qwen3-1.5B` | 1.5B | ~4GB | Medium | General Purpose |
| `Qwen/Qwen3-3B` | 3B | ~8GB | Slower | High Quality |

## 🔄 Streaming Responses

Enable streaming for real-time responses:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "user", "content": "Write a short story about a robot."}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found**: Ensure you have internet connection for model download
2. **Out of memory**: Try a smaller model or increase system memory
3. **Port already in use**: Change the port with `--port` option
4. **MPS not available**: Ensure you're on Apple Silicon and have latest macOS

### Debug Mode

Start with debug logging for more information:

```bash
python -m nano_qwen3_serving --log-level debug
```

## 📚 Next Steps

- **[Installation Guide](installation.md)**: Detailed installation instructions
- **[Configuration](configuration.md)**: Advanced configuration options
- **[API Reference](../user-guide/api-reference.md)**: Complete API documentation
- **[Examples](../examples/basic-examples.md)**: More usage examples

## 🆘 Need Help?

- Check the [Troubleshooting](../troubleshooting/common-issues.md) guide
- Open an [Issue](https://github.com/hsliuustc/nano-qwen3-serving/issues) on GitHub
- Join our [Discussions](https://github.com/hsliuustc/nano-qwen3-serving/discussions)

---

**🎉 Congratulations! You've successfully set up Nano Qwen3 Serving!** 