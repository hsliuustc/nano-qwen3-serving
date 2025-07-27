# Model Acceleration Guide

Nano Qwen3 Serving provides comprehensive model acceleration techniques to optimize inference performance and memory usage. This guide explains how to use the acceleration features effectively.

## Overview

The acceleration system supports multiple optimization techniques:

- **Quantization**: INT8, INT4, FP16, BF16 precision reduction
- **Efficient Kernels**: Flash Attention, Fused operations
- **Memory Optimizations**: Gradient checkpointing, Memory layout optimization
- **Compilation**: torch.compile with different modes
- **Benchmarking**: Performance measurement and comparison tools

## Quick Start

### Basic Usage

```python
from nano_qwen3_serving.core.model_runner import ModelRunner
from nano_qwen3_serving.core.acceleration import create_acceleration_config

# Create acceleration configuration
config = create_acceleration_config(
    quantization="fp16",
    use_flash_attention=True,
    use_torch_compile=False
)

# Initialize ModelRunner with acceleration
runner = ModelRunner(
    model_name="Qwen/Qwen3-0.6B",
    device="auto",
    acceleration_config=config
)
```

### Default Acceleration

If no configuration is provided, ModelRunner uses sensible defaults:

```python
# Uses default acceleration (Flash Attention enabled, no quantization)
runner = ModelRunner(model_name="Qwen/Qwen3-0.6B")
```

## Quantization Types

### FP16 (Half Precision)
- **Memory Reduction**: ~50%
- **Speed**: Faster on modern GPUs
- **Use Case**: Good balance of speed and accuracy

```python
config = create_acceleration_config(quantization="fp16")
```

### BF16 (Brain Float 16)
- **Memory Reduction**: ~50%
- **Speed**: Similar to FP16, better numerical stability
- **Use Case**: Recommended for training and inference

```python
config = create_acceleration_config(quantization="bf16")
```

### Dynamic INT8
- **Memory Reduction**: ~50%
- **Speed**: Faster inference, some accuracy loss
- **Use Case**: Production deployment with memory constraints

```python
config = create_acceleration_config(quantization="dynamic_int8")
```

### Static INT8
- **Memory Reduction**: ~75%
- **Speed**: Fastest inference, requires calibration
- **Use Case**: Maximum performance optimization

```python
config = create_acceleration_config(quantization="static_int8")
```

### INT4 (Experimental)
- **Memory Reduction**: ~75%
- **Speed**: Maximum compression
- **Use Case**: Extreme memory constraints

```python
config = create_acceleration_config(quantization="int4")
```

## Advanced Configuration

### Complete Configuration

```python
from nano_qwen3_serving.core.acceleration import AccelerationConfig, QuantizationType

config = AccelerationConfig(
    quantization=QuantizationType.FP16,
    use_flash_attention=True,
    use_torch_compile=True,
    compile_mode="reduce-overhead",  # or "max-autotune"
    use_channels_last=True,
    use_gradient_checkpointing=True,
    use_fused_kernels=True
)

runner = ModelRunner(
    model_name="your-model",
    acceleration_config=config
)
```

### Device-Specific Optimization

```python
# CUDA optimized
cuda_config = create_acceleration_config(
    quantization="fp16",
    use_flash_attention=True,
    use_torch_compile=True,
    compile_mode="max-autotune"
)

# CPU optimized
cpu_config = create_acceleration_config(
    quantization="dynamic_int8",
    use_flash_attention=False,
    use_torch_compile=False
)

# MPS (Apple Silicon) optimized
mps_config = create_acceleration_config(
    quantization="fp16",
    use_flash_attention=True,
    use_torch_compile=False
)
```

## Performance Monitoring

### Get Model Information

```python
info = runner.get_model_info()
print(f"Applied optimizations: {info['acceleration']['applied_optimizations']}")
print(f"Memory usage: {info['memory_usage']}")
print(f"Quantization info: {info['quantization']}")
```

### Benchmark Performance

```python
# Run performance benchmark
results = runner.benchmark_performance(
    sample_text="Hello, how are you?",
    num_iterations=100
)

print(f"Average inference time: {results['avg_inference_time']:.4f}s")
print(f"Tokens per second: {results['tokens_per_second']:.2f}")
print(f"Applied optimizations: {results['applied_optimizations']}")
```

### Memory Usage Estimation

```python
memory_info = runner.estimate_memory_usage()
print(f"Base model size: {memory_info['base_model_size_mb']:.1f} MB")
print(f"Accelerated size: {memory_info['accelerated_model_size_mb']:.1f} MB")
print(f"Memory saved: {memory_info['memory_saved_mb']:.1f} MB")
print(f"Reduction: {memory_info['memory_reduction_percent']:.1f}%")
```

## Best Practices

### For Development
```python
# Fast iteration with good debugging
config = create_acceleration_config(
    quantization="none",
    use_flash_attention=True,
    use_torch_compile=False
)
```

### For Production (GPU)
```python
# Maximum performance on GPU
config = create_acceleration_config(
    quantization="fp16",
    use_flash_attention=True,
    use_torch_compile=True,
    compile_mode="max-autotune"
)
```

### For Production (CPU)
```python
# Optimized for CPU deployment
config = create_acceleration_config(
    quantization="dynamic_int8",
    use_flash_attention=False,
    use_torch_compile=False
)
```

### For Memory-Constrained Environments
```python
# Maximum memory savings
config = create_acceleration_config(
    quantization="static_int8",
    use_flash_attention=True,
    use_gradient_checkpointing=True
)
```

## Dynamic Configuration Updates

You can update acceleration configuration at runtime:

```python
# Initial configuration
runner = ModelRunner(model_name="your-model")

# Update to use quantization
new_config = create_acceleration_config(
    quantization="fp16",
    use_torch_compile=True
)

# This will reload the model with new optimizations
runner.update_acceleration_config(new_config)
```

## Performance Estimates

The system provides estimates for optimization benefits:

| Optimization | Memory Reduction | Speed Improvement |
|--------------|------------------|-------------------|
| FP16/BF16 Quantization | ~50% | 1.2-1.5x |
| Dynamic INT8 | ~50% | 1.3-1.8x |
| Static INT8 | ~75% | 1.5-2.5x |
| Flash Attention 2 | - | 1.5-2.0x |
| Fused Kernels | - | 1.1-1.3x |
| torch.compile | - | 1.3-1.8x |
| Gradient Checkpointing | ~30% | - |

## Troubleshooting

### Common Issues

1. **torch.compile fails**: Disable with `use_torch_compile=False`
2. **Quantization errors**: Try FP16 instead of INT8
3. **Memory issues**: Enable gradient checkpointing
4. **Slow performance**: Check device compatibility

### Compatibility

- **Flash Attention**: Requires compatible GPU and model architecture
- **torch.compile**: Requires PyTorch 2.0+
- **Quantization**: Some models may not support all quantization types
- **Fused Kernels**: Model-dependent availability

### Validation

Always benchmark your specific use case:

```python
# Compare configurations
configs = [
    create_acceleration_config(quantization="none"),
    create_acceleration_config(quantization="fp16"),
    create_acceleration_config(quantization="dynamic_int8"),
]

for i, config in enumerate(configs):
    runner.update_acceleration_config(config)
    results = runner.benchmark_performance(num_iterations=50)
    print(f"Config {i}: {results['avg_inference_time']:.4f}s avg")
```

## API Reference

For detailed API documentation, see:
- `AccelerationConfig` class
- `ModelAccelerator` class  
- `create_acceleration_config()` function
- `ModelRunner.benchmark_performance()` method
- `ModelRunner.get_acceleration_info()` method