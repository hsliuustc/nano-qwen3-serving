#!/usr/bin/env python3
"""
Integration example showing ModelRunner with acceleration in a simple inference scenario.

This example demonstrates how the enhanced ModelRunner with acceleration features
can be used in practical inference scenarios without requiring actual model downloads.
"""

import sys
import os
import torch
import time
from unittest.mock import Mock

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_qwen3_serving.core.acceleration import (
    AccelerationConfig, 
    QuantizationType, 
    create_acceleration_config,
    ModelAccelerator
)


def create_demo_model():
    """Create a simple demo model for testing acceleration."""
    import torch.nn as nn
    
    class DemoTransformer(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=512, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=2048,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            self.ln_final = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            
            # Add config for acceleration compatibility
            self.config = Mock()
            self.config.use_flash_attention_2 = False
            self.config.use_memory_efficient_attention = False
            self.config.use_fused_rms_norm = False
            self.config.use_fused_mlp = False
        
        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.ln_final(x)
            logits = self.lm_head(x)
            
            # Return dict to mimic transformers models
            return type('ModelOutput', (), {
                'logits': logits,
                'past_key_values': None
            })()
        
        def num_parameters(self):
            return sum(p.numel() for p in self.parameters())
        
        def gradient_checkpointing_enable(self):
            """Mock gradient checkpointing."""
            pass
    
    return DemoTransformer()


def demo_acceleration_comparison():
    """Demonstrate different acceleration configurations."""
    print("🚀 Model Acceleration Integration Demo")
    print("=" * 50)
    
    # Create demo model
    model = create_demo_model()
    device = torch.device("cpu")  # Use CPU for demo
    
    print(f"\nDemo model created:")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Device: {device}")
    
    # Test different acceleration configurations
    configs = {
        "baseline": AccelerationConfig(
            quantization=QuantizationType.NONE,
            use_flash_attention=False,
            use_torch_compile=False,
            use_gradient_checkpointing=False,
            use_fused_kernels=False
        ),
        "fp16": create_acceleration_config(
            quantization="fp16",
            use_flash_attention=True,
            use_torch_compile=False
        ),
        "optimized": create_acceleration_config(
            quantization="fp16",
            use_flash_attention=True,
            use_torch_compile=False,
            use_gradient_checkpointing=True,
            use_fused_kernels=True
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n--- Testing {config_name.upper()} configuration ---")
        
        # Apply acceleration
        accelerator = ModelAccelerator(config)
        accelerated_model = accelerator.apply_accelerations(model, device)
        
        # Show applied optimizations
        optimizations = accelerator.applied_optimizations
        print(f"Applied optimizations: {', '.join(optimizations) if optimizations else 'None'}")
        
        # Get memory and performance estimates
        memory_reductions = accelerator.get_memory_footprint_reduction()
        performance_estimates = accelerator.get_performance_estimates()
        
        if memory_reductions:
            print("Memory reductions:")
            for opt, reduction in memory_reductions.items():
                print(f"  {opt}: {reduction:.1f}%")
        
        if performance_estimates:
            print("Performance estimates:")
            for opt, speedup in performance_estimates.items():
                print(f"  {opt}: {speedup:.1f}x speedup")
        
        # Simple benchmark
        print("Running benchmark...")
        sample_input = torch.randint(0, 1000, (1, 32))  # Batch size 1, sequence length 32
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = accelerated_model(sample_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                output = accelerated_model(sample_input)
                times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        tokens_per_second = sample_input.shape[1] / avg_time
        
        results[config_name] = {
            'avg_time': avg_time,
            'tokens_per_second': tokens_per_second,
            'optimizations': len(optimizations),
            'memory_reduction': sum(memory_reductions.values()) if memory_reductions else 0,
            'estimated_speedup': max(performance_estimates.values()) if performance_estimates else 1.0
        }
        
        print(f"Benchmark results:")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Tokens/second: {tokens_per_second:.2f}")
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("ACCELERATION COMPARISON SUMMARY")
    print("=" * 50)
    
    baseline_time = results['baseline']['avg_time']
    
    for config_name, result in results.items():
        speedup = baseline_time / result['avg_time']
        print(f"\n{config_name.upper()}:")
        print(f"  Optimizations applied: {result['optimizations']}")
        print(f"  Estimated memory reduction: {result['memory_reduction']:.1f}%")
        print(f"  Actual speedup vs baseline: {speedup:.2f}x")
        print(f"  Estimated speedup: {result['estimated_speedup']:.2f}x")
        print(f"  Tokens/second: {result['tokens_per_second']:.2f}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"\nKey Benefits Demonstrated:")
    print(f"  📈 Performance optimization through multiple techniques")
    print(f"  🔧 Configurable acceleration options")
    print(f"  📊 Built-in benchmarking and monitoring")
    print(f"  🔍 Memory usage estimation")
    print(f"  ⚡ Real performance measurements")


def demo_dynamic_config_update():
    """Demonstrate dynamic configuration updates."""
    print("\n" + "=" * 50)
    print("DYNAMIC CONFIGURATION UPDATE DEMO")
    print("=" * 50)
    
    model = create_demo_model()
    device = torch.device("cpu")
    
    # Start with baseline
    config1 = AccelerationConfig(quantization=QuantizationType.NONE, use_flash_attention=False)
    accelerator1 = ModelAccelerator(config1)
    model1 = accelerator1.apply_accelerations(model, device)
    
    print(f"\nInitial configuration:")
    print(f"  Optimizations: {', '.join(accelerator1.applied_optimizations) or 'None'}")
    
    # Update to optimized configuration
    config2 = create_acceleration_config(
        quantization="fp16",
        use_flash_attention=True,
        use_fused_kernels=True
    )
    accelerator2 = ModelAccelerator(config2)
    model2 = accelerator2.apply_accelerations(model, device)
    
    print(f"\nUpdated configuration:")
    print(f"  Optimizations: {', '.join(accelerator2.applied_optimizations)}")
    
    print(f"\n✅ Configuration update successful!")


if __name__ == "__main__":
    try:
        demo_acceleration_comparison()
        demo_dynamic_config_update()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()