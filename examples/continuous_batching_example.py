#!/usr/bin/env python3
"""
Example demonstrating continuous batching functionality.
"""

import time
from nano_qwen3_serving import LLM, SamplingParams


def demonstrate_continuous_batching():
    """Demonstrate continuous batching vs static batching."""
    print("🚀 Continuous Batching Demo")
    print("=" * 50)
    
    # Test prompts
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms",
        "Write a haiku about programming",
        "What is the capital of France?",
        "How does photosynthesis work?"
    ]
    
    # Test both batching modes
    for batching_mode in ["static", "continuous"]:
        print(f"\n📊 Testing {batching_mode.upper()} Batching")
        print("-" * 30)
        
        # Initialize LLM with specific batching mode
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            dtype="float16",
            batching_mode=batching_mode,
            max_batch_size=4,
            enable_optimizations=True
        )
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=20,
            do_sample=True
        )
        
        # Measure generation time
        start_time = time.time()
        results = llm.generate(prompts, sampling_params)
        total_time = time.time() - start_time
        
        # Print results
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📝 Generated {len(results)} responses")
        
        for i, result in enumerate(results):
            if "error" not in result:
                print(f"  {i+1}. {result['generated_text'][:50]}...")
            else:
                print(f"  {i+1}. Error: {result['error']}")
        
        # Get statistics
        stats = llm.get_stats()
        print(f"📈 Engine stats: {stats.get('total_requests', 0)} requests processed")
        
        # Shutdown
        llm.shutdown()
    
    print(f"\n✅ Demo completed!")


def demonstrate_batch_state_objects():
    """Demonstrate the structured batch state objects."""
    print("\n🔧 Batch State Objects Demo")
    print("=" * 50)
    
    from nano_qwen3_serving.core.batch_state import BatchState, BatchUpdate, SequenceInfo
    from nano_qwen3_serving.core.sampling_params import SamplingParams
    import torch
    
    # Create sampling parameters
    sampling_params = SamplingParams(max_tokens=10)
    
    # Create sequence info
    seq_info = SequenceInfo(
        sequence_id=1,
        request_id=100,
        start_position=0,
        current_length=5,
        max_new_tokens=10,
        sampling_params=sampling_params
    )
    
    print(f"📋 Sequence Info:")
    print(f"  ID: {seq_info.sequence_id}")
    print(f"  Request ID: {seq_info.request_id}")
    print(f"  Position: {seq_info.start_position}")
    print(f"  Current Length: {seq_info.current_length}")
    print(f"  Max New Tokens: {seq_info.max_new_tokens}")
    
    # Create batch state
    input_ids = torch.zeros((2, 10), dtype=torch.long)
    attention_mask = torch.ones((2, 10), dtype=torch.long)
    
    sequence_map = {1: seq_info}
    position_to_sequence = {0: 1}
    
    batch_state = BatchState(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sequence_map=sequence_map,
        position_to_sequence=position_to_sequence,
        batch_size=1,
        max_seq_length=10,
        active_sequences=[1]
    )
    
    print(f"\n📦 Batch State:")
    print(f"  Batch Size: {batch_state.batch_size}")
    print(f"  Max Seq Length: {batch_state.max_seq_length}")
    print(f"  Active Sequences: {batch_state.active_sequences}")
    print(f"  Input Shape: {batch_state.input_ids.shape}")
    
    # Create batch update
    batch_update = BatchUpdate(
        new_tokens={1: [123, 456]},
        completed_sequences=[],
        inference_time=0.1,
        tokens_generated=2
    )
    
    print(f"\n🔄 Batch Update:")
    print(f"  New Tokens: {batch_update.new_tokens}")
    print(f"  Completed Sequences: {batch_update.completed_sequences}")
    print(f"  Inference Time: {batch_update.inference_time}s")
    print(f"  Tokens Generated: {batch_update.tokens_generated}")


def demonstrate_scheduler():
    """Demonstrate the continuous batching scheduler."""
    print("\n🎯 Continuous Batching Scheduler Demo")
    print("=" * 50)
    
    from nano_qwen3_serving.core.continuous_batching_scheduler import ContinuousBatchingScheduler, Request
    from nano_qwen3_serving.core.sampling_params import SamplingParams
    
    # Create scheduler
    scheduler = ContinuousBatchingScheduler(
        max_queue_size=100,
        max_batch_size=4
    )
    
    print(f"📊 Scheduler initialized with max_batch_size={scheduler.max_batch_size}")
    
    # Add some requests
    sampling_params = SamplingParams(max_tokens=10)
    
    for i in range(3):
        request = Request(
            request_id=i,
            prompt=f"Test prompt {i}",
            sampling_params=sampling_params
        )
        scheduler.add_request(request)
    
    print(f"📝 Added {len(scheduler.pending_requests)} requests to queue")
    
    # Get initial stats
    stats = scheduler.get_stats()
    print(f"📈 Initial stats: {stats}")
    
    # Simulate adding sequences to batch
    for i in range(2):
        seq_info = SequenceInfo(
            sequence_id=i,
            request_id=i,
            start_position=i,
            current_length=0,
            max_new_tokens=10,
            sampling_params=sampling_params
        )
        scheduler.active_sequences[i] = seq_info
    
    # Get batch state
    batch_state = scheduler.get_batch_state()
    if batch_state:
        print(f"🔄 Batch state created with {batch_state.batch_size} sequences")
    
    # Update batch
    from nano_qwen3_serving.core.batch_state import BatchUpdate
    
    batch_update = BatchUpdate(
        new_tokens={0: [123], 1: [456]},
        completed_sequences=[],
        inference_time=0.05,
        tokens_generated=2
    )
    
    scheduler.update_batch(batch_update)
    
    # Get updated stats
    updated_stats = scheduler.get_stats()
    print(f"📈 Updated stats: {updated_stats}")
    
    # Shutdown
    scheduler.shutdown()
    print("✅ Scheduler demo completed")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_batch_state_objects()
    demonstrate_scheduler()
    demonstrate_continuous_batching() 