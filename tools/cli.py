#!/usr/bin/env python3
"""
Command-line interface for the nano LLM serving engine.
"""

import argparse
import sys
import json
from typing import List, Dict, Any, Optional
from loguru import logger

from nano_qwen3_serving import LLM, SamplingParams
from nano_qwen3_serving.core.scheduler import RequestPriority


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


def interactive_mode(llm: LLM, stream: bool = False):
    """Run interactive chat mode."""
    print("🤖 Nano Qwen3 Serving Engine - Interactive Mode")
    if stream:
        print("🔄 Streaming mode enabled")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'stats':
                print_stats(llm)
                continue
            elif user_input.lower() == 'clear':
                llm.clear_stats()
                print("📊 Statistics cleared")
                continue
            elif not user_input:
                continue
            
            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            
            if stream:
                # Streaming mode - temporarily reduce logging verbosity
                from loguru import logger
                logger.remove()
                logger.add(lambda msg: None, level="ERROR")
                
                try:
                    for result in llm.generate_stream(user_input):
                        if 'token' in result:
                            print(result['token'], end="", flush=True)
                        if result.get('finished', False):
                            break
                    print()  # New line after streaming
                finally:
                    # Restore original logging level
                    logger.remove()
                    logger.add(sys.stderr, level="INFO")
            else:
                # Non-streaming mode
                result = llm.generate_single(user_input)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(result["generated_text"])
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def batch_mode(llm: LLM, prompts: List[str], output_file: Optional[str] = None):
    """Run batch processing mode."""
    print(f"🔄 Processing {len(prompts)} prompts...")
    
    results = llm.generate(prompts)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {output_file}")
    else:
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"Prompt: {result['prompt']}")
                print(f"Generated: {result['generated_text']}")
                print(f"Tokens: {result['tokens_generated']}")


def single_mode(llm: LLM, prompt: str, sampling_params: Optional[SamplingParams] = None, stream: bool = False):
    """Run single prompt mode."""
    print(f"🎯 Processing: {prompt}")
    
    if stream:
        # Streaming mode - temporarily reduce logging verbosity
        from loguru import logger
        logger.remove()
        logger.add(lambda msg: None, level="ERROR")
        
        try:
            print("🤖 Generated: ", end="", flush=True)
            for result in llm.generate_stream(prompt, sampling_params):
                if 'token' in result:
                    print(result['token'], end="", flush=True)
                if result.get('finished', False):
                    break
            print()  # New line after streaming
            print(f"📊 Streaming completed")
        finally:
            # Restore original logging level
            logger.remove()
            logger.add(sys.stderr, level="INFO")
    else:
        # Non-streaming mode
    result = llm.generate_single(prompt, sampling_params)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"🤖 Generated: {result['generated_text']}")
        print(f"📊 Tokens generated: {result['tokens_generated']}")
        print(f"📊 Total tokens: {result['total_tokens']}")


def print_help():
    """Print help information."""
    print("\n📖 Available Commands:")
    print("  help    - Show this help")
    print("  stats   - Show system statistics")
    print("  clear   - Clear statistics")
    print("  quit    - Exit the program")
    print("\n💡 Tips:")
    print("  - Use natural language prompts")
    print("  - Try different types of questions")
    print("  - Check stats to monitor performance")


def print_stats(llm: LLM):
    """Print system statistics."""
    stats = llm.get_stats()
    model_info = llm.get_model_info()
    
    print("\n📊 System Statistics:")
    print(f"  Uptime: {stats['uptime']:.1f}s")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Completed: {stats['completed_requests']}")
    print(f"  Failed: {stats['failed_requests']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Requests/sec: {stats['requests_per_second']:.2f}")
    
    print("\n🤖 Model Information:")
    print(f"  Model: {model_info.get('model_name', 'Unknown')}")
    print(f"  Device: {model_info.get('device', 'Unknown')}")
    print(f"  Parameters: {model_info.get('num_parameters', 0):,}")
    
    print("\n💾 Memory Statistics:")
    mem_stats = stats.get('memory_stats', {})
    print(f"  Block Utilization: {mem_stats.get('utilization', 0):.1%}")
    print(f"  Allocated Blocks: {mem_stats.get('allocated_blocks', 0)}")
    print(f"  Free Blocks: {mem_stats.get('free_blocks', 0)}")


def create_sampling_params(args) -> SamplingParams:
    """Create sampling parameters from command line arguments."""
    if args.greedy:
        return SamplingParams.greedy(max_tokens=args.max_tokens)
    elif args.creative:
        return SamplingParams.creative(max_tokens=args.max_tokens)
    elif args.balanced:
        return SamplingParams.balanced(max_tokens=args.max_tokens)
    elif args.fast:
        return SamplingParams.greedy(max_tokens=50)
    else:
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens
        )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Nano Qwen3 Serving Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i                    # Interactive mode
  %(prog)s -i --stream           # Interactive mode with streaming
  %(prog)s "Hello, how are you?" # Single prompt
  %(prog)s --stream "Hello"      # Single prompt with streaming
  %(prog)s -f prompts.txt        # Batch mode from file
  %(prog)s --greedy "What is AI?" # Greedy decoding
  %(prog)s --creative "Write a story" # Creative sampling
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen3-0.6B",
        help="Model name or path (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--device", 
        default="mps",
        help="Device to use (default: mps)"
    )
    parser.add_argument(
        "--dtype", 
        default="float16",
        help="Data type (default: float16)"
    )
    
    # Mode selection
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "-f", "--file",
        help="Process prompts from file (one per line)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file for batch results (JSON format)"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling (default: -1, no limit)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    
    # Sampling presets
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding"
    )
    parser.add_argument(
        "--creative",
        action="store_true",
        help="Use creative sampling"
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced sampling"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode (50 tokens max, greedy decoding)"
    )
    
    # Other options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics and exit"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (real-time token generation)"
    )
    
    # Positional argument for single prompt
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Single prompt to process"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Initialize LLM
        logger.info(f"Initializing LLM with model: {args.model}")
        llm = LLM(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype
        )
        
        # Show stats and exit if requested
        if args.stats:
            print_stats(llm)
            return
        
        # Determine mode and run
        if args.interactive:
            interactive_mode(llm, args.stream)
        elif args.file:
            # Batch mode from file
            with open(args.file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            batch_mode(llm, prompts, args.output)
        elif args.prompt:
            # Single prompt mode
            sampling_params = create_sampling_params(args)
            single_mode(llm, args.prompt, sampling_params, args.stream)
        else:
            # Default to interactive mode
            interactive_mode(llm, args.stream)
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'llm' in locals():
            llm.shutdown()


if __name__ == "__main__":
    main() 