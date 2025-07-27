# Multi-Backend Support

Nano Qwen3 Serving now supports multiple backends for maximum flexibility and performance across different hardware configurations.

## 🎯 Supported Backends

### 1. **CUDA (NVIDIA GPUs)**
- **Hardware**: NVIDIA GPUs with CUDA support
- **Performance**: Highest performance for large models
- **Memory**: Efficient GPU memory management
- **Features**: Mixed precision, optimized kernels

### 2. **MPS (Apple Silicon)**
- **Hardware**: Apple M1/M2/M3 chips
- **Performance**: Optimized for Apple Silicon
- **Memory**: Unified memory architecture
- **Features**: Metal Performance Shaders, half precision

### 3. **CPU (Fallback)**
- **Hardware**: Any CPU with sufficient RAM
- **Performance**: Slower but universal compatibility
- **Memory**: System RAM management
- **Features**: Multi-threading, float32 precision

## 🚀 Quick Start

### Auto-Detection (Recommended)

```python
from nano_qwen3_serving import LLM

# Automatically detect and use the best available device
llm = LLM(
    model_name="Qwen/Qwen3-0.6B",
    device="auto"  # Will choose CUDA > MPS > CPU
)
```

### Manual Device Selection

```python
# Use specific device
llm = LLM(
    model_name="Qwen/Qwen3-0.6B",
    device="cuda"  # or "mps", "cpu"
)
```

### Command Line Usage

```bash
# Auto-detect device
python -m nano_qwen3_serving --device auto

# Use specific device
python -m nano_qwen3_serving --device cuda
python -m nano_qwen3_serving --device mps
python -m nano_qwen3_serving --device cpu
```


## 📊 Performance Comparison

| Backend | Qwen3-0.6B | 
|---------|------------|
| CUDA    | ~30        |
| MPS     | ~25        |


*Performance in tokens/second. Actual performance depends on hardware configuration.*

## 🛠️ Device Detection Tool

Use the built-in device detection tool to check your system:

```bash
# Check all devices
python tools/device_check.py

# Test model loading
python tools/device_check.py --model-test

# Run performance benchmark
python tools/device_check.py --benchmark

# Run all tests
python tools/device_check.py --all
```

Example output:
```
🔍 Nano Qwen3 Serving - Device Detection Tool
============================================================
🐍 Python version: 3.11.0
🔥 PyTorch version: 2.1.0
🎮 CUDA available: True
🍎 MPS available: False

📱 Testing device: auto
✅ Device: cuda
📊 Device Type: nvidia_gpu
🎮 GPU Name: NVIDIA GeForce RTX 4090
🔢 GPU Count: 1
⚡ Compute Capability: (8, 9)
💾 Total Memory: 24.00 GB
```

## ⚙️ Configuration

### Environment Variables

```bash
# Set default device
export NANO_QWEN3_DEVICE=auto

# CUDA specific
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# MPS specific
export PYTORCH_ENABLE_MPS_FALLBACK=1

# CPU specific
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Configuration File

```yaml
# config.yaml
model:
  device: "auto"  # auto, cuda, mps, cpu
  torch_dtype: "auto"  # auto, float16, float32, bfloat16

performance:
  # CUDA optimizations
  enable_tensor_cores: true
  enable_flash_attention: true
  cuda_memory_fraction: 0.9
  
  # MPS optimizations
  enable_half_precision: true
  mps_memory_fraction: 0.9
  
  # CPU optimizations
  num_threads: 8
  enable_compilation: false
```

## 🔍 Troubleshooting

### CUDA Issues

**CUDA out of memory:**
```bash
# Reduce batch size
python -m nano_qwen3_serving --max-batch-size 1

# Use smaller model
python -m nano_qwen3_serving --model Qwen/Qwen3-0.6B

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

**CUDA not available:**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### MPS Issues

**MPS not available:**
```bash
# Check macOS version (requires 12.3+)
sw_vers

# Check Apple Silicon
uname -m  # Should return arm64

# Fallback to CPU
python -m nano_qwen3_serving --device cpu
```

### CPU Issues

**Slow performance:**
```bash
# Increase threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Use smaller model
python -m nano_qwen3_serving --model Qwen/Qwen3-0.6B
```

## 🎯 Best Practices

### 1. **Device Selection**
- Use `device="auto"` for automatic detection
- Prefer CUDA for NVIDIA GPUs
- Use MPS for Apple Silicon
- CPU as fallback for compatibility

### 2. **Memory Management**
- Monitor memory usage with device stats
- Use appropriate model sizes for your hardware
- Clear cache when switching devices

### 3. **Performance Optimization**
- Use appropriate data types (float16 for GPU, float32 for CPU)
- Adjust batch sizes based on memory
- Monitor performance with device stats

### 4. **Development Workflow**
- Test on multiple devices during development
- Use device detection tool for debugging
- Monitor performance metrics

## 📈 Advanced Usage

### Multi-GPU Support

```python
# Use specific GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llm = LLM(device="cuda")
```

### Device Switching

```python
# Switch devices at runtime
llm = LLM(device="auto")

# Get current device info
stats = llm.get_stats()
device_info = stats['model_stats']['device_stats']
print(f"Current device: {device_info['device']}")
```

### Custom Optimizations

```python
from nano_qwen3_serving.core.device_manager import DeviceManager

# Create custom device manager
device_manager = DeviceManager("cuda")

# Apply custom optimizations
model = device_manager.optimize_for_device(model)

# Get device stats
memory_stats = device_manager.get_memory_stats()
```

## 🔄 Migration Guide

### From MPS-Only to Multi-Backend

1. **Update device parameter:**
   ```python
   # Old
   llm = LLM(device="mps")
   
   # New
   llm = LLM(device="auto")  # or "cuda", "cpu"
   ```

2. **Update command line:**
   ```bash
   # Old
   python -m nano_qwen3_serving --device mps
   
   # New
   python -m nano_qwen3_serving --device auto
   ```

3. **Test on different devices:**
   ```bash
   python tools/device_check.py --all
   ```

## 📚 Additional Resources

- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [Apple MPS Documentation](https://developer.apple.com/metal/pytorch/)
- [Performance Tuning Guide](performance-tuning.md)
- [Troubleshooting Guide](troubleshooting.md) 