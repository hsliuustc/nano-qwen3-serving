# Installation Guide

This guide covers all installation methods for Nano Qwen3 Serving, from basic pip installation to advanced development setups.

## 🎯 System Requirements

### Minimum Requirements

- **Operating System**: macOS 12.3+ (Monterey) or later
- **Hardware**: Apple Silicon (M1, M2, M3, or M1 Pro/Max/Ultra)
- **Memory**: 8GB RAM (16GB+ recommended)
- **Storage**: 5GB free space for models
- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12

### Recommended Requirements

- **Memory**: 16GB+ RAM for larger models
- **Storage**: 20GB+ free space for multiple models
- **Python**: 3.11 or 3.12
- **Network**: Stable internet connection for model downloads

## 🚀 Installation Methods

### Method 1: PyPI Installation (Recommended)

The easiest way to install Nano Qwen3 Serving:

```bash
pip install nano-qwen3-serving
```

**Verify installation:**
```bash
python -c "import nano_qwen3_serving; print('Installation successful!')"
```

### Method 2: Source Installation

For development or custom modifications:

```bash
# Clone the repository
git clone https://github.com/hsliuustc/nano-qwen3-serving.git
cd nano-qwen3-serving

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

### Method 3: Conda Installation

Using conda for environment management:

```bash
# Create a new conda environment
conda create -n nano-qwen3 python=3.11
conda activate nano-qwen3

# Install the package
pip install nano-qwen3-serving
```

### Method 4: Docker Installation

For containerized deployment:

```bash
# Pull the Docker image
docker pull hsliuustc/nano-qwen3-serving:latest

# Run the container
docker run -p 8000:8000 hsliuustc/nano-qwen3-serving:latest
```

## 📦 Dependencies

### Core Dependencies

The following packages are automatically installed:

- **torch**: PyTorch with MPS support
- **transformers**: Hugging Face transformers library
- **fastapi**: Web framework for API
- **uvicorn**: ASGI server
- **pydantic**: Data validation
- **loguru**: Logging framework

### Optional Dependencies

Install additional features:

```bash
# Development dependencies
pip install nano-qwen3-serving[dev]

# Testing dependencies
pip install nano-qwen3-serving[test]

# All dependencies
pip install nano-qwen3-serving[all]
```

## 🔧 Environment Setup

### 1. Python Environment

**Using venv (recommended):**
```bash
# Create virtual environment
python -m venv nano-qwen3-env

# Activate environment
source nano-qwen3-env/bin/activate  # macOS/Linux
# nano-qwen3-env\Scripts\activate  # Windows

# Install package
pip install nano-qwen3-serving
```

**Using conda:**
```bash
# Create conda environment
conda create -n nano-qwen3 python=3.11

# Activate environment
conda activate nano-qwen3

# Install package
pip install nano-qwen3-serving
```

### 2. Environment Variables

Set optional environment variables:

```bash
# Server configuration
export NANO_QWEN3_PORT=8000
export NANO_QWEN3_HOST=127.0.0.1
export NANO_QWEN3_MODEL=Qwen/Qwen3-0.6B
export NANO_QWEN3_DEVICE=mps

# Logging
export NANO_QWEN3_LOG_LEVEL=info

# Hugging Face
export HUGGING_FACE_HUB_TOKEN=your_token_here
export HF_HOME=./models  # Custom model cache directory
```

### 3. Model Cache Configuration

Configure where models are stored:

```bash
# Set Hugging Face cache directory
export HF_HOME=./models

# Or use Python
import os
os.environ['HF_HOME'] = './models'
```

## 🍎 Apple Silicon Setup

### 1. Verify Apple Silicon

Check if you have Apple Silicon:

```bash
# Check processor architecture
uname -m
# Should return: arm64

# Check processor type
sysctl -n machdep.cpu.brand_string
# Should contain: Apple M1, M2, or M3
```

### 2. Install PyTorch with MPS Support

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 3. Test MPS Functionality

```python
import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available!")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")
```

## 🔍 Verification

### 1. Basic Verification

Test the installation:

```bash
# Check if package is installed
python -c "import nano_qwen3_serving; print('Package installed successfully')"

# Check version
python -c "import nano_qwen3_serving; print(nano_qwen3_serving.__version__)"
```

### 2. Model Download Test

Test model downloading:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Test small model download
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model downloaded successfully!")
```

### 3. Server Test

Test the server:

```bash
# Start server in background
python -m nano_qwen3_serving --port 8001 &

# Test health endpoint
curl http://localhost:8001/health

# Stop server
pkill -f "nano_qwen3_serving"
```

## 🛠️ Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/hsliuustc/nano-qwen3-serving.git
cd nano-qwen3-serving
```

### 2. Install Development Dependencies

```bash
# Install in development mode
pip install -e ".[dev]"

# Or install manually
pip install -e .
pip install pytest pytest-asyncio black isort mypy
```

### 3. Setup Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nano_qwen3_serving

# Run specific test file
pytest tests/test_basic.py
```

## 🐳 Docker Setup

### 1. Build Docker Image

```bash
# Build from source
docker build -t nano-qwen3-serving .

# Or pull from registry
docker pull hsliuustc/nano-qwen3-serving:latest
```

### 2. Run Container

```bash
# Basic run
docker run -p 8000:8000 nano-qwen3-serving

# With custom model
docker run -p 8000:8000 -e NANO_QWEN3_MODEL=Qwen/Qwen3-1.5B nano-qwen3-serving

# With volume for model cache
docker run -p 8000:8000 -v ./models:/app/models nano-qwen3-serving
```

### 3. Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  nano-qwen3:
    image: hsliuustc/nano-qwen3-serving:latest
    ports:
      - "8000:8000"
    environment:
      - NANO_QWEN3_MODEL=Qwen/Qwen3-0.6B
      - NANO_QWEN3_DEVICE=mps
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## 🔧 Troubleshooting Installation

### Common Issues

1. **Permission Errors:**
   ```bash
   # Use user installation
   pip install --user nano-qwen3-serving
   ```

2. **PyTorch MPS Issues:**
   ```bash
   # Reinstall PyTorch
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio
   ```

3. **Model Download Issues:**
   ```bash
   # Set Hugging Face token
   export HUGGING_FACE_HUB_TOKEN=your_token
   
   # Or use mirror
   export HF_ENDPOINT=https://hf-mirror.com
   ```

4. **Memory Issues:**
   ```bash
   # Use smaller model
   python -m nano_qwen3_serving --model Qwen/Qwen3-0.6B
   ```

### Getting Help

- Check [Troubleshooting Guide](../troubleshooting/common-issues.md)
- Open an [Issue](https://github.com/hsliuustc/nano-qwen3-serving/issues)
- Join [Discussions](https://github.com/hsliuustc/nano-qwen3-serving/discussions)

## 📚 Next Steps

After installation:

1. **[Quick Start](quick-start.md)**: Get up and running in minutes
2. **[Configuration](configuration.md)**: Configure the server
3. **[API Reference](../user-guide/api-reference.md)**: Learn the API
4. **[Examples](../examples/basic-examples.md)**: See usage examples

---

**🎉 Installation complete! You're ready to start using Nano Qwen3 Serving!** 