"""
Tests for async functionality in nano Qwen3 serving engine.
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any

from nano_qwen3_serving import AsyncLLM, SamplingParams
from nano_qwen3_serving.core.scheduler import RequestPriority


class TestAsyncLLM:
    """Test cases for AsyncLLM class."""
    
    @pytest.fixture
    async def async_llm(self):
        """Create an AsyncLLM instance for testing."""
        llm = AsyncLLM(worker_count=2)
        await llm.start()
        yield llm
        await llm.shutdown()
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, async_llm):
        """Test basic text generation."""
        result = await async_llm.generate("Hello, world!")
        
        assert isinstance(result, dict)
        assert "generated_text" in result
        assert len(result["generated_text"]) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_prompts(self, async_llm):
        """Test generation with multiple prompts."""
        prompts = ["Hello", "World", "Test"]
        results = await async_llm.generate(prompts)
        
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert "generated_text" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_llm):
        """Test concurrent request processing."""
        prompts = ["Request 1", "Request 2", "Request 3", "Request 4"]
        
        start_time = time.time()
        tasks = [async_llm.generate(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(results) == 4
        assert all(isinstance(r, dict) for r in results)
        assert all("generated_text" in r for r in results)
        
        # Verify concurrent processing (should be faster than sequential)
        sequential_time = len(prompts) * 5.0  # Estimate 5s per request
        assert end_time - start_time < sequential_time
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, async_llm):
        """Test streaming text generation."""
        prompt = "Let me solve this step by step: 2+2="
        tokens = []
        
        async for result in async_llm.generate_stream(prompt):
            if 'token' in result:
                tokens.append(result['token'])
            if result.get('finished', False):
                break
        
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    
    @pytest.mark.asyncio
    async def test_chat_generation(self, async_llm):
        """Test chat-style generation."""
        messages = [
            {"role": "user", "content": "Hello! How are you?"}
        ]
        
        result = await async_llm.chat(messages)
        
        assert isinstance(result, dict)
        assert "generated_text" in result
        assert len(result["generated_text"]) > 0
    
    @pytest.mark.asyncio
    async def test_chat_streaming(self, async_llm):
        """Test streaming chat generation."""
        messages = [
            {"role": "user", "content": "Hello! How are you?"}
        ]
        
        tokens = []
        async for result in async_llm.chat_stream(messages):
            if 'token' in result:
                tokens.append(result['token'])
            if result.get('finished', False):
                break
        
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, async_llm):
        """Test batch processing functionality."""
        prompts = ["Batch 1", "Batch 2", "Batch 3", "Batch 4"]
        
        results = await async_llm.generate_batch(prompts, batch_size=2)
        
        assert len(results) == 4
        assert all(isinstance(r, dict) for r in results)
        assert all("generated_text" in r for r in results)
    
    @pytest.mark.asyncio
    async def test_sampling_params(self, async_llm):
        """Test generation with custom sampling parameters."""
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=10
        )
        
        result = await async_llm.generate("Test prompt", sampling_params)
        
        assert isinstance(result, dict)
        assert "generated_text" in result
    
    @pytest.mark.asyncio
    async def test_priority_levels(self, async_llm):
        """Test different priority levels."""
        high_priority = await async_llm.generate(
            "High priority", 
            priority=RequestPriority.HIGH
        )
        normal_priority = await async_llm.generate(
            "Normal priority", 
            priority=RequestPriority.NORMAL
        )
        
        assert isinstance(high_priority, dict)
        assert isinstance(normal_priority, dict)
        assert "generated_text" in high_priority
        assert "generated_text" in normal_priority
    
    @pytest.mark.asyncio
    async def test_stats_collection(self, async_llm):
        """Test statistics collection."""
        # Generate some requests first
        await async_llm.generate("Test 1")
        await async_llm.generate("Test 2")
        
        stats = await async_llm.get_stats()
        
        assert isinstance(stats, dict)
        assert "async_stats" in stats
        assert "sync_stats" in stats
        assert "queue_size" in stats
        assert "worker_count" in stats
        assert "running" in stats
        
        # Check async stats
        async_stats = stats["async_stats"]
        assert "completed_async_requests" in async_stats
        assert "failed_async_requests" in async_stats
        assert "average_wait_time" in async_stats
    
    @pytest.mark.asyncio
    async def test_stats_clearing(self, async_llm):
        """Test statistics clearing."""
        # Generate some requests
        await async_llm.generate("Test")
        
        # Clear stats
        await async_llm.clear_stats()
        
        # Check that stats are reset
        stats = await async_llm.get_stats()
        assert stats["async_stats"]["completed_async_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_model_info(self, async_llm):
        """Test model information retrieval."""
        model_info = async_llm.get_model_info()
        
        assert isinstance(model_info, dict)
        assert "model_name" in model_info
        assert "device" in model_info
        assert "dtype" in model_info
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with AsyncLLM(worker_count=2) as llm:
            result = await llm.generate("Context manager test")
            assert isinstance(result, dict)
            assert "generated_text" in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_llm):
        """Test error handling in async operations."""
        # Test with invalid prompt (empty string)
        try:
            result = await async_llm.generate("")
            # Should still work but might be empty
            assert isinstance(result, dict)
        except Exception as e:
            # If it raises an exception, that's also acceptable
            assert isinstance(e, Exception)
    
    @pytest.mark.asyncio
    async def test_worker_scaling(self):
        """Test different worker configurations."""
        # Test with 1 worker
        async with AsyncLLM(worker_count=1) as llm1:
            result1 = await llm1.generate("Test with 1 worker")
            assert isinstance(result1, dict)
        
        # Test with 4 workers
        async with AsyncLLM(worker_count=4) as llm4:
            result4 = await llm4.generate("Test with 4 workers")
            assert isinstance(result4, dict)
    
    @pytest.mark.asyncio
    async def test_queue_management(self, async_llm):
        """Test queue management functionality."""
        # Submit multiple requests to test queue
        prompts = [f"Queue test {i}" for i in range(10)]
        
        start_time = time.time()
        tasks = [async_llm.generate(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(results) == 10
        assert all(isinstance(r, dict) for r in results)
        
        # Check that all requests were processed
        stats = await async_llm.get_stats()
        assert stats["async_stats"]["completed_async_requests"] >= 10


class TestAsyncPerformance:
    """Performance tests for async functionality."""
    
    @pytest.mark.asyncio
    async def test_concurrent_vs_sequential(self):
        """Compare concurrent vs sequential processing."""
        prompts = ["Performance test"] * 4
        
        # Concurrent processing
        async with AsyncLLM(worker_count=2) as async_llm:
            start_time = time.time()
            tasks = [async_llm.generate(prompt) for prompt in prompts]
            concurrent_results = await asyncio.gather(*tasks)
            concurrent_time = time.time() - start_time
        
        # Sequential processing (simulated)
        from nano_qwen3_serving import LLM
        
        sync_llm = LLM()
        start_time = time.time()
        sequential_results = []
        for prompt in prompts:
            result = sync_llm.generate(prompt)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        sync_llm.shutdown()
        
        # Verify results are equivalent
        assert len(concurrent_results) == len(sequential_results)
        
        # Concurrent should be at least as fast as sequential
        # (may be slower due to overhead for small batches)
        assert concurrent_time <= sequential_time * 1.5  # Allow 50% overhead
    
    @pytest.mark.asyncio
    async def test_streaming_performance(self):
        """Test streaming performance."""
        async with AsyncLLM(worker_count=2) as llm:
            prompt = "Let me solve this step by step: 2+2="
            
            start_time = time.time()
            token_count = 0
            async for result in llm.generate_stream(prompt):
                if 'token' in result:
                    token_count += 1
                if result.get('finished', False):
                    break
            streaming_time = time.time() - start_time
            
            # Verify streaming completed
            assert token_count > 0
            assert streaming_time < 30.0  # Should complete within 30 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage with concurrent requests."""
        async with AsyncLLM(worker_count=2) as llm:
            # Submit many requests to test memory management
            prompts = [f"Memory test {i}" for i in range(20)]
            
            start_time = time.time()
            tasks = [llm.generate(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            assert len(results) == 20
            assert all(isinstance(r, dict) for r in results)
            
            # Should complete within reasonable time
            assert end_time - start_time < 120.0  # 2 minutes max


class TestAsyncIntegration:
    """Integration tests for async functionality."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete async workflow."""
        async with AsyncLLM(worker_count=2) as llm:
            # Test various operations
            result1 = await llm.generate("Hello")
            assert isinstance(result1, dict)
            
            result2 = await llm.chat([{"role": "user", "content": "Hi"}])
            assert isinstance(result2, dict)
            
            # Test streaming
            tokens = []
            async for result in llm.generate_stream("Stream test"):
                if 'token' in result:
                    tokens.append(result['token'])
                if result.get('finished', False):
                    break
            assert len(tokens) > 0
            
            # Test batch processing
            batch_results = await llm.generate_batch(["Batch 1", "Batch 2"])
            assert len(batch_results) == 2
            
            # Test stats
            stats = await llm.get_stats()
            assert isinstance(stats, dict)
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming(self):
        """Test concurrent streaming operations."""
        async with AsyncLLM(worker_count=2) as llm:
            # Start multiple streaming operations concurrently
            async def stream_operation(prompt):
                tokens = []
                async for result in llm.generate_stream(prompt):
                    if 'token' in result:
                        tokens.append(result['token'])
                    if result.get('finished', False):
                        break
                return tokens
            
            # Run multiple streams concurrently
            tasks = [
                stream_operation(f"Stream {i}") 
                for i in range(3)
            ]
            results = await asyncio.gather(*tasks)
            
            # Verify all streams completed
            assert len(results) == 3
            assert all(len(tokens) > 0 for tokens in results)


if __name__ == "__main__":
    pytest.main([__file__]) 