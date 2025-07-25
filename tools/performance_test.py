#!/usr/bin/env python3
"""
Performance testing script for nano Qwen3 serving engine.
"""

import time
import json
import argparse

from nano_qwen3_serving import LLM, SamplingParams


def run_performance_test(device: str = "mps"):
    """Run comprehensive performance tests."""
    print("🚀 Nano Qwen3 Serving Engine - Performance Test")
    print("=" * 60)
    
    # Initialize LLM
    print(f"📥 Initializing LLM...(device={device})")
    start_time = time.time()
    llm = LLM(
        model_name= "/zx_data1/nano-vllm/models/Qwen3-0.6B", # "Qwen/Qwen3-0.6B",
        device=device,
        dtype="float16"
    )
    init_time = time.time() - start_time
    print(f"✅ LLM initialized in {init_time:.2f}s")
    
    # Test prompts
    test_prompts = [
        "What is AI?",
        "Explain quantum computing",
        "Write a haiku about coding",
        "What is the capital of France?",
        "How does photosynthesis work?"
    ]
    
    # Test different token limits
    token_limits = [10, 20, 50, 100]
    
    results = []
    
    for max_tokens in token_limits:
        print(f"\n🔍 Testing with max_tokens={max_tokens}")
        print("-" * 40)
        
        for i, prompt in enumerate(test_prompts):
            print(f"  {i+1}. Testing: '{prompt}'")
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=max_tokens,
                do_sample=True
            )
            
            # Generate response
            start_time = time.time()
            result = llm.generate_single(prompt, sampling_params)
            generation_time = time.time() - start_time
            
            if "error" not in result:
                tokens_generated = result["tokens_generated"]
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                
                print(f"     ✅ Generated {tokens_generated} tokens in {generation_time:.2f}s")
                print(f"     📊 Speed: {tokens_per_second:.1f} tokens/sec")
                print(f"     💬 Response: {result['generated_text'][:100]}...")
                
                results.append({
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "tokens_generated": tokens_generated,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "response_preview": result['generated_text'][:100]
                })
            else:
                print(f"     ❌ Error: {result['error']}")
    
    # Get final statistics
    print(f"\n📊 Final System Statistics")
    print("=" * 60)
    
    stats = llm.get_stats()
    model_info = llm.get_model_info()
    
    print(f"🤖 Model Information:")
    print(f"  Model: {model_info.get('model_name', 'Unknown')}")
    print(f"  Device: {model_info.get('device', 'Unknown')}")
    print(f"  Parameters: {model_info.get('num_parameters', 0):,}")
    print(f"  Data Type: {model_info.get('dtype', 'Unknown')}")
    
    print(f"\n📈 Performance Summary:")
    if results:
        avg_tokens_per_sec = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_generation_time = sum(r["generation_time"] for r in results) / len(results)
        total_tokens = sum(r["tokens_generated"] for r in results)
        
        print(f"  Average Speed: {avg_tokens_per_sec:.1f} tokens/sec")
        print(f"  Average Generation Time: {avg_generation_time:.2f}s")
        print(f"  Total Tokens Generated: {total_tokens}")
        print(f"  Total Test Time: {sum(r['generation_time'] for r in results):.2f}s")
    
    print(f"\n💾 Memory Statistics:")
    mem_stats = stats.get('memory_stats', {})
    print(f"  Block Utilization: {mem_stats.get('utilization', 0):.1%}")
    print(f"  Allocated Blocks: {mem_stats.get('allocated_blocks', 0)}")
    print(f"  Free Blocks: {mem_stats.get('free_blocks', 0)}")
    
    print(f"\n🔧 Model Performance:")
    model_stats = stats.get('model_stats', {})
    print(f"  Average Inference Time: {model_stats.get('average_inference_time', 0):.4f}s")
    print(f"  Total Inference Time: {model_stats.get('total_inference_time', 0):.2f}s")
    print(f"  Inference Count: {model_stats.get('inference_count', 0)}")
    print(f"  Tokens per Second: {model_stats.get('tokens_per_second', 0):.1f}")
    
    # Save detailed results
    detailed_results = {
        "test_config": {
            "model": model_info.get('model_name'),
            "device": model_info.get('device'),
            "parameters": model_info.get('num_parameters'),
            "init_time": init_time
        },
        "performance_summary": {
            "avg_tokens_per_sec": avg_tokens_per_sec if results else 0,
            "avg_generation_time": avg_generation_time if results else 0,
            "total_tokens": total_tokens if results else 0,
            "total_test_time": sum(r['generation_time'] for r in results) if results else 0
        },
        "detailed_results": results,
        "system_stats": stats
    }
    
    with open("performance_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: performance_results.json")
    
    # Shutdown
    llm.shutdown()
    print(f"\n✅ Performance test completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano Qwen3 Serving Engine performance test")
    parser.add_argument("--device", type=str, default="mps", help="device (e.g. cpu, cuda, mps, auto)")
    args = parser.parse_args()
    run_performance_test(device=args.device) 