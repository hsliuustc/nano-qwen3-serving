#!/usr/bin/env python3
"""
Device detection and configuration tool for nano Qwen3 serving engine.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nano_qwen3_serving.core.device_manager import DeviceManager


def check_devices():
    """Check all available devices and their configurations."""
    print("🔍 Device Detection and Configuration")
    print("=" * 50)
    
    # Test different device configurations
    devices_to_test = ["auto", "cuda", "mps", "cpu"]
    
    for device_name in devices_to_test:
        print(f"\n📱 Testing device: {device_name}")
        print("-" * 30)
        
        try:
            device_manager = DeviceManager(device_name)
            
            print(f"✅ Device: {device_manager.device}")
            print(f"📊 Device Type: {device_manager.device_info['device_type']}")
            
            if device_manager.device == "cuda":
                print(f"🎮 GPU Name: {device_manager.device_info['gpu_name']}")
                print(f"🔢 GPU Count: {device_manager.device_info['gpu_count']}")
                print(f"⚡ Compute Capability: {device_manager.device_info['compute_capability']}")
                
                memory_info = device_manager.device_info['memory_info']
                print(f"💾 Total Memory: {memory_info['total'] / (1024**3):.2f} GB")
                print(f"📈 Allocated Memory: {memory_info['allocated'] / (1024**3):.2f} GB")
                print(f"🗄️  Cached Memory: {memory_info['cached'] / (1024**3):.2f} GB")
                
            elif device_manager.device == "mps":
                print(f"🍎 GPU Name: {device_manager.device_info['gpu_name']}")
                print(f"✅ MPS Available: {device_manager.device_info['mps_available']}")
                
            else:  # CPU
                memory_info = device_manager.device_info['memory_info']
                print(f"💾 Total Memory: {memory_info['total'] / (1024**3):.2f} GB")
                print(f"📈 Used Memory: {memory_info['used'] / (1024**3):.2f} GB")
                print(f"📊 Available Memory: {memory_info['available'] / (1024**3):.2f} GB")
                print(f"📈 Memory Usage: {memory_info['percent']:.1f}%")
            
            print(f"🎯 Optimal Data Type: {device_manager.get_dtype()}")
            print(f"⚙️  Optimization Config: {device_manager.optimization_config}")
            
        except Exception as e:
            print(f"❌ Error testing {device_name}: {e}")


def test_model_loading():
    """Test model loading on different devices."""
    print("\n🧠 Model Loading Test")
    print("=" * 30)
    
    # Test with a small model
    model_name = "Qwen/Qwen3-0.6B"
    
    for device_name in ["auto", "cuda", "mps", "cpu"]:
        print(f"\n📱 Testing model loading on {device_name}")
        print("-" * 30)
        
        try:
            from nano_qwen3_serving.core.model_runner import ModelRunner
            
            print(f"🔄 Loading model: {model_name}")
            model_runner = ModelRunner(
                model_name=model_name,
                device=device_name,
                max_seq_length=512  # Smaller for testing
            )
            
            print(f"✅ Model loaded successfully on {model_runner.device}")
            print(f"📊 Model parameters: {model_runner.model.num_parameters():,}")
            
            # Test basic inference
            print("🧪 Testing basic inference...")
            test_input = "Hello, world!"
            input_ids = model_runner.tokenize(test_input)
            
            with torch.no_grad():
                outputs = model_runner.run_model(input_ids)
            
            print(f"✅ Inference successful")
            print(f"📈 Output shape: {outputs['logits'].shape}")
            print(f"⏱️  Inference time: {outputs['inference_time']:.4f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def performance_benchmark():
    """Run performance benchmark on available devices."""
    print("\n🚀 Performance Benchmark")
    print("=" * 30)
    
    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms",
        "Write a haiku about programming",
    ]
    
    for device_name in ["auto", "cuda", "mps", "cpu"]:
        print(f"\n📱 Benchmarking {device_name}")
        print("-" * 30)
        
        try:
            from nano_qwen3_serving import LLM, SamplingParams
            
            # Initialize LLM
            llm = LLM(
                model_name="Qwen/Qwen3-0.6B",
                device=device_name,
                max_seq_length=256  # Smaller for faster testing
            )
            
            # Test parameters
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=20,
                do_sample=True
            )
            
            # Run benchmark
            import time
            start_time = time.time()
            
            results = llm.generate(test_prompts, sampling_params)
            
            total_time = time.time() - start_time
            
            print(f"✅ Completed {len(results)} generations")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"📊 Average time per prompt: {total_time/len(results):.2f}s")
            print(f"🚀 Tokens per second: {sum(len(r.get('generated_text', '').split()) for r in results) / total_time:.1f}")
            
            # Get device stats
            stats = llm.get_stats()
            if 'model_stats' in stats and 'device_stats' in stats['model_stats']:
                device_stats = stats['model_stats']['device_stats']
                print(f"💾 Device memory: {device_stats}")
            
            llm.shutdown()
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Device detection and configuration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Check all devices
  %(prog)s --model-test       # Test model loading
  %(prog)s --benchmark         # Run performance benchmark
  %(prog)s --all              # Run all tests
        """
    )
    
    parser.add_argument(
        "--model-test",
        action="store_true",
        help="Test model loading on different devices"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    # Import torch here to avoid import errors
    import torch
    
    print("🔍 Nano Qwen3 Serving - Device Detection Tool")
    print("=" * 60)
    
    # Check PyTorch version and CUDA availability
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    print(f"🍎 MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 CUDA version: {torch.version.cuda}")
        print(f"🎮 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Run tests based on arguments
    if args.all or not any([args.model_test, args.benchmark]):
        check_devices()
    
    if args.all or args.model_test:
        test_model_loading()
    
    if args.all or args.benchmark:
        performance_benchmark()
    
    print("\n✅ Device detection completed!")


if __name__ == "__main__":
    main() 