#!/usr/bin/env python3
"""
Demo script showing model acceleration features.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_qwen3_serving.core.acceleration import (
    AccelerationConfig, 
    QuantizationType, 
    create_acceleration_config
)

def main():
    """Demo the acceleration configuration."""
    print("🚀 Nano Qwen3 Serving - Model Acceleration Demo")
    print("=" * 50)
    
    # Demo 1: Create basic acceleration config
    print("\n1. Basic Acceleration Configuration:")
    config1 = create_acceleration_config()
    print(f"   Default quantization: {config1.quantization.value}")
    print(f"   Flash attention: {config1.use_flash_attention}")
    print(f"   Torch compile: {config1.use_torch_compile}")
    
    # Demo 2: Create FP16 quantization config
    print("\n2. FP16 Quantization Configuration:")
    config2 = create_acceleration_config(
        quantization="fp16",
        use_flash_attention=True,
        use_torch_compile=False
    )
    print(f"   Quantization: {config2.quantization.value}")
    print(f"   Flash attention: {config2.use_flash_attention}")
    
    # Demo 3: Create INT8 quantization config
    print("\n3. Dynamic INT8 Quantization Configuration:")
    config3 = create_acceleration_config(
        quantization="dynamic_int8",
        use_flash_attention=True,
        use_torch_compile=True
    )
    print(f"   Quantization: {config3.quantization.value}")
    print(f"   Flash attention: {config3.use_flash_attention}")
    print(f"   Torch compile: {config3.use_torch_compile}")
    
    # Demo 4: Show all quantization types
    print("\n4. Available Quantization Types:")
    for qtype in QuantizationType:
        print(f"   - {qtype.value}")
    
    # Demo 5: Configuration details
    print("\n5. Advanced Configuration:")
    config4 = AccelerationConfig(
        quantization=QuantizationType.FP16,
        use_flash_attention=True,
        use_torch_compile=True,
        compile_mode="max-autotune",
        use_channels_last=True,
        use_gradient_checkpointing=True,
        use_fused_kernels=True
    )
    print(f"   Quantization: {config4.quantization.value}")
    print(f"   Compile mode: {config4.compile_mode}")
    print(f"   Channels last: {config4.use_channels_last}")
    print(f"   Gradient checkpointing: {config4.use_gradient_checkpointing}")
    print(f"   Fused kernels: {config4.use_fused_kernels}")
    
    print("\n✅ Acceleration configuration demo completed!")
    print("\nTo use these configurations with ModelRunner:")
    print("```python")
    print("from nano_qwen3_serving.core.model_runner import ModelRunner")
    print("from nano_qwen3_serving.core.acceleration import create_acceleration_config")
    print("")
    print("config = create_acceleration_config(quantization='fp16', use_flash_attention=True)")
    print("runner = ModelRunner(model_name='your-model', acceleration_config=config)")
    print("```")

if __name__ == "__main__":
    main()