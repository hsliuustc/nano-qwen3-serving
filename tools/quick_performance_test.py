#!/usr/bin/env python3
"""
Quick performance test for continuous batching vs static batching.
"""

import time
import statistics
from typing import List, Dict, Any
from loguru import logger

from nano_qwen3_serving import LLM, SamplingParams


def quick_performance_test():
    """Run a quick performance comparison between batching modes."""
    print("🚀 Quick Performance Test: Continuous vs Static Batching")
    print("=" * 60)
    
    # Test configuration
    model_name = "Qwen/Qwen3-0.6B"
    device = "mps"
    batch_sizes = [1, 2, 4]
    num_requests = 10
    
    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms",
        "Write a haiku about programming",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is machine learning?",
        "Explain blockchain technology",
        "What are renewable energy sources?",
        "How do vaccines work?",
        "What is climate change?"
    ]
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=30,
        do_sample=True
    )
    
    results = []
    
    # Test both batching modes
    for batching_mode in ["static", "continuous"]:
        print(f"\n📊 Testing {batching_mode.upper()} Batching")
        print("-" * 40)
        
        mode_results = []
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Initialize LLM
            start_init = time.time()
            llm = LLM(
                model_name=model_name,
                device=device,
                dtype="float16",
                batching_mode=batching_mode,
                max_batch_size=batch_size,
                enable_optimizations=True
            )
            init_time = time.time() - start_init
            
            # Measure generation
            start_time = time.time()
            generation_results = llm.generate(test_prompts[:num_requests], sampling_params)
            total_time = time.time() - start_time
            
            # Calculate metrics
            avg_latency = total_time / num_requests
            throughput = num_requests / total_time
            
            # Calculate tokens per second
            total_tokens = sum(r.get('tokens_generated', 0) for r in generation_results if 'error' not in r)
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            
            # Get system stats
            stats = llm.get_stats()
            
            result = {
                'batching_mode': batching_mode,
                'batch_size': batch_size,
                'num_requests': num_requests,
                'init_time': init_time,
                'total_time': total_time,
                'avg_latency': avg_latency,
                'throughput': throughput,
                'tokens_per_second': tokens_per_second,
                'total_tokens': total_tokens,
                'stats': stats
            }
            
            mode_results.append(result)
            results.append(result)
            
            print(f"    ⏱️  Total time: {total_time:.2f}s")
            print(f"    📊 Throughput: {throughput:.2f} req/s")
            print(f"    🎯 Avg latency: {avg_latency:.3f}s")
            print(f"    🔤 Tokens/sec: {tokens_per_second:.1f}")
            print(f"    📝 Total tokens: {total_tokens}")
            
            # Shutdown
            llm.shutdown()
            
            # Small delay between runs
            time.sleep(1)
        
        # Calculate averages for this mode
        avg_throughput = statistics.mean([r['throughput'] for r in mode_results])
        avg_latency = statistics.mean([r['avg_latency'] for r in mode_results])
        avg_tokens_per_sec = statistics.mean([r['tokens_per_second'] for r in mode_results])
        
        print(f"\n  📈 {batching_mode.upper()} Batching Averages:")
        print(f"    Throughput: {avg_throughput:.2f} req/s")
        print(f"    Latency: {avg_latency:.3f}s")
        print(f"    Tokens/sec: {avg_tokens_per_sec:.1f}")
    
    # Compare results
    print(f"\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    static_results = [r for r in results if r['batching_mode'] == 'static']
    continuous_results = [r for r in results if r['batching_mode'] == 'continuous']
    
    if static_results and continuous_results:
        # Calculate average improvements
        static_throughput = statistics.mean([r['throughput'] for r in static_results])
        continuous_throughput = statistics.mean([r['throughput'] for r in continuous_results])
        throughput_improvement = ((continuous_throughput - static_throughput) / static_throughput) * 100
        
        static_latency = statistics.mean([r['avg_latency'] for r in static_results])
        continuous_latency = statistics.mean([r['avg_latency'] for r in continuous_results])
        latency_improvement = ((static_latency - continuous_latency) / static_latency) * 100
        
        static_tps = statistics.mean([r['tokens_per_second'] for r in static_results])
        continuous_tps = statistics.mean([r['tokens_per_second'] for r in continuous_results])
        tps_improvement = ((continuous_tps - static_tps) / static_tps) * 100
        
        print(f"\n📈 Performance Improvements with Continuous Batching:")
        print(f"  Throughput: {throughput_improvement:+.1f}% ({static_throughput:.2f} → {continuous_throughput:.2f} req/s)")
        print(f"  Latency: {latency_improvement:+.1f}% ({static_latency:.3f}s → {continuous_latency:.3f}s)")
        print(f"  Tokens/sec: {tps_improvement:+.1f}% ({static_tps:.1f} → {continuous_tps:.1f})")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if throughput_improvement > 5:
            print(f"  ✅ Continuous batching provides {throughput_improvement:.1f}% higher throughput")
        else:
            print(f"  ⚠️  Throughput improvement is minimal ({throughput_improvement:.1f}%)")
        
        if latency_improvement > 5:
            print(f"  ✅ Continuous batching reduces latency by {latency_improvement:.1f}%")
        else:
            print(f"  ⚠️  Latency improvement is minimal ({latency_improvement:.1f}%)")
        
        if tps_improvement > 5:
            print(f"  ✅ Continuous batching improves token generation by {tps_improvement:.1f}%")
        else:
            print(f"  ⚠️  Token generation improvement is minimal ({tps_improvement:.1f}%)")
        
        # Overall recommendation
        total_improvement = (throughput_improvement + latency_improvement + tps_improvement) / 3
        if total_improvement > 10:
            print(f"\n🎯 Overall: Continuous batching shows significant improvement ({total_improvement:.1f}% average)")
        elif total_improvement > 5:
            print(f"\n🎯 Overall: Continuous batching shows moderate improvement ({total_improvement:.1f}% average)")
        else:
            print(f"\n🎯 Overall: Continuous batching shows minimal improvement ({total_improvement:.1f}% average)")
    
    # Detailed results table
    print(f"\n📋 Detailed Results:")
    print(f"{'Mode':<12} {'Batch':<6} {'Throughput':<12} {'Latency':<10} {'Tokens/s':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['batching_mode']:<12} {result['batch_size']:<6} "
              f"{result['throughput']:<12.2f} {result['avg_latency']:<10.3f} "
              f"{result['tokens_per_second']:<10.1f}")
    
    print(f"\n✅ Quick performance test completed!")


def test_concurrent_vs_sequential():
    """Test concurrent vs sequential request processing."""
    print(f"\n🔄 Testing Concurrent vs Sequential Processing")
    print("=" * 50)
    
    # Test configuration
    model_name = "Qwen/Qwen3-0.6B"
    device = "mps"
    num_requests = 5
    
    # Test prompts
    test_prompts = [
        "What is AI?",
        "Explain ML",
        "What is Python?",
        "How does HTTP work?",
        "What is JSON?"
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=20,
        do_sample=True
    )
    
    # Test sequential processing (static batching)
    print("📊 Sequential Processing (Static Batching)")
    print("-" * 40)
    
    start_time = time.time()
    llm_static = LLM(
        model_name=model_name,
        device=device,
        dtype="float16",
        batching_mode="static",
        max_batch_size=1,  # Force sequential
        enable_optimizations=True
    )
    
    static_results = llm_static.generate(test_prompts, sampling_params)
    static_time = time.time() - start_time
    llm_static.shutdown()
    
    print(f"  ⏱️  Total time: {static_time:.2f}s")
    print(f"  📊 Throughput: {num_requests/static_time:.2f} req/s")
    
    # Test concurrent processing (continuous batching)
    print("\n📊 Concurrent Processing (Continuous Batching)")
    print("-" * 40)
    
    start_time = time.time()
    llm_continuous = LLM(
        model_name=model_name,
        device=device,
        dtype="float16",
        batching_mode="continuous",
        max_batch_size=num_requests,  # Allow full concurrency
        enable_optimizations=True
    )
    
    continuous_results = llm_continuous.generate(test_prompts, sampling_params)
    continuous_time = time.time() - start_time
    llm_continuous.shutdown()
    
    print(f"  ⏱️  Total time: {continuous_time:.2f}s")
    print(f"  📊 Throughput: {num_requests/continuous_time:.2f} req/s")
    
    # Compare
    speedup = static_time / continuous_time if continuous_time > 0 else 0
    print(f"\n📈 Speedup: {speedup:.2f}x faster with continuous batching")
    
    if speedup > 1.5:
        print("✅ Continuous batching provides significant concurrency benefits")
    elif speedup > 1.1:
        print("✅ Continuous batching provides moderate concurrency benefits")
    else:
        print("⚠️  Continuous batching provides minimal concurrency benefits")


def main():
    """Run the quick performance tests."""
    print("🚀 Quick Performance Tests")
    print("=" * 50)
    
    # Run quick performance test
    quick_performance_test()
    
    # Run concurrent vs sequential test
    test_concurrent_vs_sequential()
    
    print(f"\n🎉 All performance tests completed!")


if __name__ == "__main__":
    main() 