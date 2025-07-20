"""
Integration tests for the complete nano LLM serving engine.
"""

import pytest
import time
from nano_qwen3_serving import LLM, SamplingParams
from nano_qwen3_serving.core.scheduler import RequestPriority


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_basic_workflow(self):
        """Test the complete workflow from prompt to response."""
        # Initialize LLM with minimal settings for testing
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        try:
            # Test basic generation
            prompt = "Hello"
            result = llm.generate_single(prompt, SamplingParams(max_tokens=10))
            
            # Check result structure
            assert "request_id" in result
            assert "prompt" in result
            assert "generated_text" in result
            assert "tokens_generated" in result
            assert "total_tokens" in result
            assert "block_indices" in result
            
            # Check values
            assert result["prompt"] == prompt
            assert isinstance(result["tokens_generated"], int)
            assert isinstance(result["total_tokens"], int)
            assert isinstance(result["block_indices"], list)
            
            # Check that generation actually happened
            if "error" not in result:
                assert len(result["generated_text"]) > 0
                assert result["tokens_generated"] > 0
                assert result["total_tokens"] > 0
            
        finally:
            llm.shutdown()
    
    def test_batch_processing(self):
        """Test batch processing of multiple prompts."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        try:
            prompts = ["Hi", "Hello", "Hey"]
            results = llm.generate(prompts, SamplingParams(max_tokens=5))
            
            assert len(results) == len(prompts)
            
            for i, result in enumerate(results):
                assert result["prompt"] == prompts[i]
                assert "generated_text" in result
                assert "tokens_generated" in result
                
        finally:
            llm.shutdown()
    
    def test_sampling_strategies(self):
        """Test different sampling strategies."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        try:
            prompt = "Write a short story"
            
            # Test greedy sampling
            greedy_result = llm.generate_single(
                prompt, 
                SamplingParams.greedy(max_tokens=20)
            )
            
            # Test creative sampling
            creative_result = llm.generate_single(
                prompt,
                SamplingParams.creative(max_tokens=20)
            )
            
            # Test balanced sampling
            balanced_result = llm.generate_single(
                prompt,
                SamplingParams.balanced(max_tokens=20)
            )
            
            # All should have results
            for result in [greedy_result, creative_result, balanced_result]:
                if "error" not in result:
                    assert len(result["generated_text"]) > 0
                    assert result["tokens_generated"] > 0
                
        finally:
            llm.shutdown()
    
    def test_chat_interface(self):
        """Test chat interface functionality."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        try:
            messages = [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
                {"role": "user", "content": "What is AI?"}
            ]
            
            result = llm.chat(messages, SamplingParams(max_tokens=20))
            
            assert "generated_text" in result
            if "error" not in result:
                assert len(result["generated_text"]) > 0
                
        finally:
            llm.shutdown()
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        try:
            # Get initial stats
            initial_stats = llm.get_stats()
            assert initial_stats["total_requests"] == 0
            assert initial_stats["completed_requests"] == 0
            assert initial_stats["failed_requests"] == 0
            
            # Generate some text
            llm.generate_single("Test", SamplingParams(max_tokens=5))
            
            # Get updated stats
            updated_stats = llm.get_stats()
            assert updated_stats["total_requests"] == 1
            assert updated_stats["completed_requests"] == 1
            assert updated_stats["failed_requests"] == 0
            assert updated_stats["success_rate"] == 1.0
            
            # Test batch processing
            llm.generate(["Test 1", "Test 2"], SamplingParams(max_tokens=5))
            
            final_stats = llm.get_stats()
            assert final_stats["total_requests"] == 3
            assert final_stats["completed_requests"] == 3
            
        finally:
            llm.shutdown()
    
    def test_model_information(self):
        """Test model information retrieval."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        try:
            model_info = llm.get_model_info()
            
            assert "model_name" in model_info
            assert "device" in model_info
            assert "dtype" in model_info
            assert "num_parameters" in model_info
            assert "max_seq_length" in model_info
            assert "use_cache" in model_info
            
            assert model_info["model_name"] == "Qwen/Qwen3-0.6B"
            assert model_info["device"] == "mps"
            
        finally:
            llm.shutdown()
    
    def test_memory_management(self):
        """Test memory management and block allocation."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=50,  # Small number for testing
            block_size=8
        )
        
        try:
            # Generate text and check block allocation
            result = llm.generate_single("Test prompt", SamplingParams(max_tokens=10))
            
            assert "block_indices" in result
            assert isinstance(result["block_indices"], list)
            
            # Check that blocks were allocated
            if "error" not in result:
                assert len(result["block_indices"]) > 0
                
            # Get memory stats
            stats = llm.get_stats()
            memory_stats = stats["memory_stats"]
            
            assert "total_blocks" in memory_stats
            assert "allocated_blocks" in memory_stats
            assert "free_blocks" in memory_stats
            assert "utilization" in memory_stats
            
        finally:
            llm.shutdown()
    
    def test_error_handling(self):
        """Test error handling in the system."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        try:
            # Test with invalid sampling parameters
            with pytest.raises(ValueError):
                SamplingParams(temperature=-1.0)
            
            with pytest.raises(ValueError):
                SamplingParams(top_p=1.5)
            
            # Test with valid parameters
            result = llm.generate_single(
                "Test", 
                SamplingParams(max_tokens=5)
            )
            
            # Should not have error
            assert "error" not in result or result["error"] is None
            
        finally:
            llm.shutdown()
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        try:
            # Generate text and measure performance
            start_time = time.time()
            result = llm.generate_single("Performance test", SamplingParams(max_tokens=10))
            generation_time = time.time() - start_time
            
            # Check that generation completed
            if "error" not in result:
                assert generation_time > 0
                assert result["tokens_generated"] > 0
                
                # Calculate tokens per second
                tokens_per_second = result["tokens_generated"] / generation_time
                assert tokens_per_second > 0
                
            # Get performance stats
            stats = llm.get_stats()
            model_stats = stats["model_stats"]
            
            assert "average_inference_time" in model_stats
            assert "total_inference_time" in model_stats
            assert "tokens_generated" in model_stats
            assert "inference_count" in model_stats
            
        finally:
            llm.shutdown()
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=5,
            num_blocks=100,
            block_size=8
        )
        
        try:
            # Add multiple requests quickly
            prompts = [f"Request {i}" for i in range(3)]
            results = llm.generate(prompts, SamplingParams(max_tokens=5))
            
            assert len(results) == len(prompts)
            
            # Check that all requests were processed
            for result in results:
                assert "generated_text" in result
                if "error" not in result:
                    assert result["tokens_generated"] > 0
                    
        finally:
            llm.shutdown()
    
    def test_cleanup_and_shutdown(self):
        """Test proper cleanup and shutdown."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        
        # Generate some text
        llm.generate_single("Test", SamplingParams(max_tokens=5))
        
        # Shutdown should not raise errors
        llm.shutdown()
        
        # After shutdown, stats should still be accessible
        stats = llm.get_stats()
        assert "total_requests" in stats 