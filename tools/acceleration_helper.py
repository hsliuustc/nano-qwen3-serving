#!/usr/bin/env python3
"""
Acceleration configuration helper utility.

This script helps users choose the best acceleration configuration for their specific use case.
"""

import sys
import os
import argparse
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_qwen3_serving.core.acceleration import (
    AccelerationConfig, 
    QuantizationType, 
    create_acceleration_config
)


def get_device_recommendations() -> Dict[str, Dict[str, Any]]:
    """Get device-specific acceleration recommendations."""
    return {
        "cuda": {
            "description": "NVIDIA GPU with CUDA support",
            "recommended_config": {
                "quantization": "fp16",
                "use_flash_attention": True,
                "use_torch_compile": True,
                "compile_mode": "max-autotune",
                "use_gradient_checkpointing": True,
                "use_fused_kernels": True
            },
            "benefits": [
                "Maximum performance with Flash Attention",
                "FP16 quantization for 2x memory savings",
                "torch.compile for additional speedup",
                "Fused kernels for efficiency"
            ],
            "considerations": [
                "Requires modern GPU with compute capability 7.0+",
                "Flash Attention needs sufficient GPU memory",
                "torch.compile has warmup overhead"
            ]
        },
        "mps": {
            "description": "Apple Silicon (M1/M2/M3) with Metal Performance Shaders",
            "recommended_config": {
                "quantization": "fp16",
                "use_flash_attention": True,
                "use_torch_compile": False,
                "use_gradient_checkpointing": True,
                "use_fused_kernels": False
            },
            "benefits": [
                "Optimized for Apple Silicon architecture",
                "FP16 quantization supported natively",
                "Good memory efficiency"
            ],
            "considerations": [
                "torch.compile may not be stable on MPS",
                "Some fused kernels not available",
                "Flash Attention support varies"
            ]
        },
        "cpu": {
            "description": "CPU-only deployment",
            "recommended_config": {
                "quantization": "dynamic_int8",
                "use_flash_attention": False,
                "use_torch_compile": False,
                "use_gradient_checkpointing": True,
                "use_fused_kernels": False
            },
            "benefits": [
                "Dynamic INT8 quantization for memory savings",
                "No GPU memory requirements",
                "Stable across different hardware"
            ],
            "considerations": [
                "Slower than GPU inference",
                "Flash Attention not beneficial on CPU",
                "Quantization provides main speedup"
            ]
        }
    }


def get_use_case_recommendations() -> Dict[str, Dict[str, Any]]:
    """Get use case-specific acceleration recommendations."""
    return {
        "development": {
            "description": "Development and debugging",
            "recommended_config": {
                "quantization": "none",
                "use_flash_attention": True,
                "use_torch_compile": False,
                "use_gradient_checkpointing": False,
                "use_fused_kernels": True
            },
            "benefits": [
                "Full precision for accurate debugging",
                "Fast iteration with minimal optimization overhead",
                "Some optimizations for reasonable performance"
            ]
        },
        "production_performance": {
            "description": "Production deployment prioritizing speed",
            "recommended_config": {
                "quantization": "fp16",
                "use_flash_attention": True,
                "use_torch_compile": True,
                "compile_mode": "max-autotune",
                "use_gradient_checkpointing": False,
                "use_fused_kernels": True
            },
            "benefits": [
                "Maximum inference speed",
                "Balanced memory usage",
                "All performance optimizations enabled"
            ]
        },
        "production_memory": {
            "description": "Production deployment prioritizing memory efficiency",
            "recommended_config": {
                "quantization": "static_int8",
                "use_flash_attention": True,
                "use_torch_compile": True,
                "use_gradient_checkpointing": True,
                "use_fused_kernels": True
            },
            "benefits": [
                "Maximum memory savings (up to 75%)",
                "Gradient checkpointing reduces activation memory",
                "Still maintains good performance"
            ]
        },
        "edge_deployment": {
            "description": "Edge devices with limited resources",
            "recommended_config": {
                "quantization": "dynamic_int8",
                "use_flash_attention": False,
                "use_torch_compile": False,
                "use_gradient_checkpointing": True,
                "use_fused_kernels": False
            },
            "benefits": [
                "Minimal memory footprint",
                "CPU-friendly optimizations",
                "Stable performance on various hardware"
            ]
        },
        "research": {
            "description": "Research and experimentation",
            "recommended_config": {
                "quantization": "none",
                "use_flash_attention": True,
                "use_torch_compile": False,
                "use_gradient_checkpointing": True,
                "use_fused_kernels": True
            },
            "benefits": [
                "Full precision for research accuracy",
                "Memory efficiency for larger experiments",
                "Flexible optimization options"
            ]
        }
    }


def print_configuration(config: AccelerationConfig, name: str, description: str, benefits: List[str], considerations: List[str] = None):
    """Print a configuration recommendation."""
    print(f"\n{'='*60}")
    print(f"📋 {name.upper()}: {description}")
    print(f"{'='*60}")
    
    print(f"\n🔧 Recommended Configuration:")
    print(f"  quantization: {config.quantization.value}")
    print(f"  use_flash_attention: {config.use_flash_attention}")
    print(f"  use_torch_compile: {config.use_torch_compile}")
    if hasattr(config, 'compile_mode') and config.use_torch_compile:
        print(f"  compile_mode: {config.compile_mode}")
    print(f"  use_gradient_checkpointing: {config.use_gradient_checkpointing}")
    print(f"  use_fused_kernels: {config.use_fused_kernels}")
    
    print(f"\n✅ Benefits:")
    for benefit in benefits:
        print(f"  • {benefit}")
    
    if considerations:
        print(f"\n⚠️  Considerations:")
        for consideration in considerations:
            print(f"  • {consideration}")
    
    print(f"\n💻 Code Example:")
    print(f"```python")
    print(f"from nano_qwen3_serving.core.acceleration import create_acceleration_config")
    print(f"from nano_qwen3_serving.core.model_runner import ModelRunner")
    print(f"")
    print(f"config = create_acceleration_config(")
    print(f"    quantization='{config.quantization.value}',")
    print(f"    use_flash_attention={config.use_flash_attention},")
    print(f"    use_torch_compile={config.use_torch_compile}")
    if hasattr(config, 'compile_mode') and config.use_torch_compile:
        print(f"    compile_mode='{config.compile_mode}'")
    print(f")")
    print(f"")
    print(f"runner = ModelRunner(")
    print(f"    model_name='your-model-name',")
    print(f"    acceleration_config=config")
    print(f")")
    print(f"```")


def main():
    """Main function for the acceleration helper."""
    parser = argparse.ArgumentParser(
        description="Acceleration Configuration Helper for Nano Qwen3 Serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python acceleration_helper.py --device cuda
  python acceleration_helper.py --use-case production_performance
  python acceleration_helper.py --list-all
  python acceleration_helper.py --compare
        """
    )
    
    parser.add_argument(
        '--device', 
        choices=['cuda', 'mps', 'cpu'], 
        help='Get recommendations for specific device type'
    )
    
    parser.add_argument(
        '--use-case', 
        choices=['development', 'production_performance', 'production_memory', 'edge_deployment', 'research'],
        help='Get recommendations for specific use case'
    )
    
    parser.add_argument(
        '--list-all', 
        action='store_true',
        help='Show all available recommendations'
    )
    
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='Compare different quantization options'
    )
    
    args = parser.parse_args()
    
    print("🚀 Nano Qwen3 Serving - Acceleration Configuration Helper")
    print("=" * 60)
    
    if args.device:
        device_recs = get_device_recommendations()
        rec = device_recs[args.device]
        config = create_acceleration_config(**rec['recommended_config'])
        print_configuration(
            config, 
            f"{args.device.upper()} Device",
            rec['description'],
            rec['benefits'],
            rec.get('considerations', [])
        )
    
    elif args.use_case:
        use_case_recs = get_use_case_recommendations()
        rec = use_case_recs[args.use_case]
        config = create_acceleration_config(**rec['recommended_config'])
        print_configuration(
            config,
            args.use_case.replace('_', ' ').title(),
            rec['description'],
            rec['benefits']
        )
    
    elif args.list_all:
        print("\n📱 DEVICE-SPECIFIC RECOMMENDATIONS")
        device_recs = get_device_recommendations()
        for device_name, rec in device_recs.items():
            config = create_acceleration_config(**rec['recommended_config'])
            print_configuration(
                config,
                f"{device_name.upper()} Device", 
                rec['description'],
                rec['benefits'][:2]  # Show only first 2 benefits for brevity
            )
        
        print("\n\n🎯 USE CASE-SPECIFIC RECOMMENDATIONS")
        use_case_recs = get_use_case_recommendations()
        for use_case_name, rec in use_case_recs.items():
            config = create_acceleration_config(**rec['recommended_config'])
            print_configuration(
                config,
                use_case_name.replace('_', ' ').title(),
                rec['description'],
                rec['benefits'][:2]  # Show only first 2 benefits for brevity
            )
    
    elif args.compare:
        print("\n📊 QUANTIZATION COMPARISON")
        print("=" * 60)
        
        quantization_info = {
            "none": {"memory_reduction": "0%", "speed": "Baseline", "accuracy": "Full precision"},
            "fp16": {"memory_reduction": "~50%", "speed": "1.2-1.5x", "accuracy": "Minimal loss"},
            "bf16": {"memory_reduction": "~50%", "speed": "1.2-1.5x", "accuracy": "Better than FP16"},
            "dynamic_int8": {"memory_reduction": "~50%", "speed": "1.3-1.8x", "accuracy": "Small loss"},
            "static_int8": {"memory_reduction": "~75%", "speed": "1.5-2.5x", "accuracy": "Moderate loss"},
            "int4": {"memory_reduction": "~75%", "speed": "2.0-3.0x", "accuracy": "Significant loss"}
        }
        
        print(f"{'Quantization':<15} {'Memory':<12} {'Speed':<12} {'Accuracy':<15}")
        print("-" * 60)
        
        for quant_type, info in quantization_info.items():
            print(f"{quant_type:<15} {info['memory_reduction']:<12} {info['speed']:<12} {info['accuracy']:<15}")
        
        print(f"\n💡 Recommendations:")
        print(f"  • Development: none or fp16")
        print(f"  • Production GPU: fp16 or bf16")
        print(f"  • Production CPU: dynamic_int8")
        print(f"  • Memory-constrained: static_int8 or int4")
    
    else:
        print("\nPlease specify an option. Use --help for available commands.")
        print("\nQuick examples:")
        print("  --device cuda          # GPU recommendations")
        print("  --use-case development # Development setup")
        print("  --list-all            # Show all options")
        print("  --compare             # Compare quantization types")


if __name__ == "__main__":
    main()