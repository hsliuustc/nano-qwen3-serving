#!/usr/bin/env python3
"""
Performance benchmarking tool for continuous batching vs static batching.
"""

import time
import json
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from nano_qwen3_serving import LLM, SamplingParams


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    batching_mode: str
    batch_size: int
    num_requests: int
    total_time: float
    avg_latency: float
    throughput: float  # requests per second
    tokens_per_second: float
    gpu_utilization: float
    memory_usage: Dict[str, float]
    latency_percentiles: Dict[str, float]


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for batching strategies."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str = "mps"):
        """
        Initialize the performance benchmark.
        
        Args:
            model_name: Model to benchmark
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        self.results = []
        
        # Test configurations
        self.batch_sizes = [1, 2, 4, 8]
        self.request_counts = [10, 20, 50, 100]
        self.prompt_lengths = [10, 50, 100, 200]  # words
        
        logger.info(f"Performance benchmark initialized for {model_name} on {device}")
    
    def generate_test_prompts(self, count: int, length: int) -> List[str]:
        """Generate test prompts of specified length."""
        base_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of renewable energy sources?",
            "Describe the process of photosynthesis step by step.",
            "How does the internet work? Explain the basics.",
            "What is quantum computing and why is it important?",
            "Explain the water cycle in detail.",
            "What are the main causes of climate change?",
            "How do vaccines work to protect against diseases?",
            "Explain the concept of blockchain technology.",
            "What is artificial intelligence and how is it used today?"
        ]
        
        prompts = []
        for i in range(count):
            # Use base prompts and extend them to desired length
            base = base_prompts[i % len(base_prompts)]
            if length <= len(base.split()):
                prompts.append(" ".join(base.split()[:length]))
            else:
                # Extend the prompt
                extension = f" Please provide a detailed explanation with at least {length} words."
                prompts.append(base + extension)
        
        return prompts
    
    def measure_single_run(
        self, 
        batching_mode: str, 
        batch_size: int, 
        prompts: List[str],
        sampling_params: SamplingParams
    ) -> BenchmarkResult:
        """
        Measure performance for a single configuration.
        
        Args:
            batching_mode: "static" or "continuous"
            batch_size: Maximum batch size
            prompts: List of prompts to process
            sampling_params: Sampling parameters
            
        Returns:
            BenchmarkResult with performance metrics
        """
        logger.info(f"Testing {batching_mode} batching with batch_size={batch_size}, prompts={len(prompts)}")
        
        # Initialize LLM
        start_init = time.time()
        llm = LLM(
            model_name=self.model_name,
            device=self.device,
            dtype="float16",
            batching_mode=batching_mode,
            max_batch_size=batch_size,
            enable_optimizations=True
        )
        init_time = time.time() - start_init
        
        # Measure generation
        start_time = time.time()
        results = llm.generate(prompts, sampling_params)
        total_time = time.time() - start_time
        
        # Calculate metrics
        num_requests = len(prompts)
        avg_latency = total_time / num_requests
        throughput = num_requests / total_time
        
        # Calculate tokens per second
        total_tokens = sum(r.get('tokens_generated', 0) for r in results if 'error' not in r)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        # Get system stats
        stats = llm.get_stats()
        memory_stats = stats.get('memory_stats', {})
        
        # Calculate latency percentiles (simplified)
        latencies = [avg_latency] * num_requests  # In real implementation, track individual latencies
        latency_percentiles = {
            'p50': statistics.median(latencies),
            'p90': np.percentile(latencies, 90),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
        
        # Shutdown
        llm.shutdown()
        
        result = BenchmarkResult(
            batching_mode=batching_mode,
            batch_size=batch_size,
            num_requests=num_requests,
            total_time=total_time,
            avg_latency=avg_latency,
            throughput=throughput,
            tokens_per_second=tokens_per_second,
            gpu_utilization=0.0,  # Would need GPU monitoring
            memory_usage=memory_stats,
            latency_percentiles=latency_percentiles
        )
        
        logger.info(f"Completed: {throughput:.2f} req/s, {tokens_per_second:.1f} tokens/s")
        return result
    
    def run_comprehensive_benchmark(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmark across all configurations."""
        logger.info("Starting comprehensive performance benchmark")
        
        # Standard sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
            do_sample=True
        )
        
        results = []
        
        # Test different batch sizes
        for batch_size in self.batch_sizes:
            for num_requests in self.request_counts:
                # Generate test prompts
                prompts = self.generate_test_prompts(num_requests, 50)
                
                # Test both batching modes
                for batching_mode in ["static", "continuous"]:
                    try:
                        result = self.measure_single_run(
                            batching_mode=batching_mode,
                            batch_size=batch_size,
                            prompts=prompts,
                            sampling_params=sampling_params
                        )
                        results.append(result)
                        
                        # Small delay between runs
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Benchmark failed for {batching_mode} batching, batch_size={batch_size}: {e}")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and calculate improvements."""
        if not self.results:
            return {}
        
        analysis = {
            'summary': {},
            'improvements': {},
            'recommendations': []
        }
        
        # Group results by configuration
        static_results = [r for r in self.results if r.batching_mode == "static"]
        continuous_results = [r for r in self.results if r.batching_mode == "continuous"]
        
        # Calculate average improvements
        if static_results and continuous_results:
            # Throughput improvement
            static_throughput = statistics.mean([r.throughput for r in static_results])
            continuous_throughput = statistics.mean([r.throughput for r in continuous_results])
            throughput_improvement = ((continuous_throughput - static_throughput) / static_throughput) * 100
            
            # Latency improvement
            static_latency = statistics.mean([r.avg_latency for r in static_results])
            continuous_latency = statistics.mean([r.avg_latency for r in continuous_results])
            latency_improvement = ((static_latency - continuous_latency) / static_latency) * 100
            
            # Tokens per second improvement
            static_tps = statistics.mean([r.tokens_per_second for r in static_results])
            continuous_tps = statistics.mean([r.tokens_per_second for r in continuous_results])
            tps_improvement = ((continuous_tps - static_tps) / static_tps) * 100
            
            analysis['summary'] = {
                'static_throughput': static_throughput,
                'continuous_throughput': continuous_throughput,
                'static_latency': static_latency,
                'continuous_latency': continuous_latency,
                'static_tokens_per_second': static_tps,
                'continuous_tokens_per_second': continuous_tps
            }
            
            analysis['improvements'] = {
                'throughput_improvement_percent': throughput_improvement,
                'latency_improvement_percent': latency_improvement,
                'tokens_per_second_improvement_percent': tps_improvement
            }
            
            # Generate recommendations
            if throughput_improvement > 10:
                analysis['recommendations'].append(
                    f"Continuous batching provides {throughput_improvement:.1f}% higher throughput"
                )
            
            if latency_improvement > 10:
                analysis['recommendations'].append(
                    f"Continuous batching reduces latency by {latency_improvement:.1f}%"
                )
            
            if tps_improvement > 10:
                analysis['recommendations'].append(
                    f"Continuous batching improves token generation by {tps_improvement:.1f}%"
                )
        
        return analysis
    
    def generate_plots(self, save_path: str = "performance_plots.png"):
        """Generate performance comparison plots."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Continuous vs Static Batching Performance Comparison', fontsize=16)
        
        # Group results
        static_results = [r for r in self.results if r.batching_mode == "static"]
        continuous_results = [r for r in self.results if r.batching_mode == "continuous"]
        
        # Plot 1: Throughput vs Batch Size
        ax1 = axes[0, 0]
        static_throughput = [r.throughput for r in static_results]
        continuous_throughput = [r.throughput for r in continuous_results]
        batch_sizes = [r.batch_size for r in static_results]
        
        ax1.plot(batch_sizes, static_throughput, 'o-', label='Static Batching', color='blue')
        ax1.plot(batch_sizes, continuous_throughput, 's-', label='Continuous Batching', color='red')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (requests/sec)')
        ax1.set_title('Throughput vs Batch Size')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Latency vs Batch Size
        ax2 = axes[0, 1]
        static_latency = [r.avg_latency for r in static_results]
        continuous_latency = [r.avg_latency for r in continuous_results]
        
        ax2.plot(batch_sizes, static_latency, 'o-', label='Static Batching', color='blue')
        ax2.plot(batch_sizes, continuous_latency, 's-', label='Continuous Batching', color='red')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Average Latency (seconds)')
        ax2.set_title('Latency vs Batch Size')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Tokens per Second vs Batch Size
        ax3 = axes[1, 0]
        static_tps = [r.tokens_per_second for r in static_results]
        continuous_tps = [r.tokens_per_second for r in continuous_results]
        
        ax3.plot(batch_sizes, static_tps, 'o-', label='Static Batching', color='blue')
        ax3.plot(batch_sizes, continuous_tps, 's-', label='Continuous Batching', color='red')
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Tokens per Second')
        ax3.set_title('Token Generation Rate vs Batch Size')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Improvement Percentage
        ax4 = axes[1, 1]
        throughput_improvement = []
        for i in range(len(static_throughput)):
            if static_throughput[i] > 0:
                improvement = ((continuous_throughput[i] - static_throughput[i]) / static_throughput[i]) * 100
                throughput_improvement.append(improvement)
        
        ax4.bar(range(len(throughput_improvement)), throughput_improvement, color='green', alpha=0.7)
        ax4.set_xlabel('Batch Size Index')
        ax4.set_ylabel('Throughput Improvement (%)')
        ax4.set_title('Throughput Improvement with Continuous Batching')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance plots saved to {save_path}")
    
    def save_results(self, filename: str = "performance_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Convert results to dictionaries
        results_dict = []
        for result in self.results:
            results_dict.append({
                'batching_mode': result.batching_mode,
                'batch_size': result.batch_size,
                'num_requests': result.num_requests,
                'total_time': result.total_time,
                'avg_latency': result.avg_latency,
                'throughput': result.throughput,
                'tokens_per_second': result.tokens_per_second,
                'gpu_utilization': result.gpu_utilization,
                'memory_usage': result.memory_usage,
                'latency_percentiles': result.latency_percentiles
            })
        
        # Add analysis
        analysis = self.analyze_results()
        
        output = {
            'model_name': self.model_name,
            'device': self.device,
            'timestamp': time.time(),
            'results': results_dict,
            'analysis': analysis
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        analysis = self.analyze_results()
        
        print("\n" + "="*60)
        print("🚀 PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        if 'summary' in analysis:
            summary = analysis['summary']
            print(f"\n📊 Average Performance:")
            print(f"  Static Batching:")
            print(f"    Throughput: {summary['static_throughput']:.2f} requests/sec")
            print(f"    Latency: {summary['static_latency']:.3f} seconds")
            print(f"    Tokens/sec: {summary['static_tokens_per_second']:.1f}")
            
            print(f"  Continuous Batching:")
            print(f"    Throughput: {summary['continuous_throughput']:.2f} requests/sec")
            print(f"    Latency: {summary['continuous_latency']:.3f} seconds")
            print(f"    Tokens/sec: {summary['continuous_tokens_per_second']:.1f}")
        
        if 'improvements' in analysis:
            improvements = analysis['improvements']
            print(f"\n📈 Performance Improvements:")
            print(f"  Throughput: +{improvements['throughput_improvement_percent']:.1f}%")
            print(f"  Latency: -{improvements['latency_improvement_percent']:.1f}%")
            print(f"  Tokens/sec: +{improvements['tokens_per_second_improvement_percent']:.1f}%")
        
        if 'recommendations' in analysis:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "="*60)


def main():
    """Run the performance benchmark."""
    print("🚀 Starting Performance Benchmark")
    print("="*50)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(
        model_name="Qwen/Qwen3-0.6B",
        device="mps"
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Analyze and save results
    benchmark.analyze_results()
    benchmark.save_results()
    benchmark.generate_plots()
    benchmark.print_summary()
    
    print("\n✅ Performance benchmark completed!")


if __name__ == "__main__":
    main() 