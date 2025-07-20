"""
Test suite for streaming functionality.
"""

import pytest
import time
from typing import List, Dict, Any

from nano_qwen3_serving import LLM, SamplingParams


class TestStreaming:
    """Test streaming generation functionality."""
    
    @pytest.fixture
    def llm(self):
        """Create LLM instance for testing."""
        llm = LLM(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            max_queue_size=10,
            num_blocks=100,
            block_size=8
        )
        yield llm
        llm.shutdown()
    
    def test_basic_streaming(self, llm):
        """Test basic streaming functionality."""
        prompt = "Hello, how are you?"
        sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
        
        # Collect streaming results
        tokens = []
        texts = []
        finished = False
        
        for result in llm.generate_stream(prompt, sampling_params):
            tokens.append(result["token"])
            texts.append(result["text"])
            finished = result["finished"]
            
            # Verify result structure
            assert "token" in result
            assert "token_id" in result
            assert "text" in result
            assert "finished" in result
            assert "tokens_generated" in result
            assert "request_id" in result
            
            if finished:
                break
        
        # Verify results
        assert len(tokens) > 0, "Should generate at least one token"
        assert finished, "Should finish generation"
        assert len(texts) == len(tokens), "Text count should match token count"
        
        # Verify accumulated text
        accumulated = "".join(tokens)
        assert texts[-1] == accumulated, "Final text should match accumulated tokens"
    
    def test_streaming_vs_non_streaming(self, llm):
        """Test that streaming and non-streaming produce same results."""
        prompt = "Explain quantum computing in simple terms."
        sampling_params = SamplingParams(max_tokens=20, temperature=0.0)
        
        # Get non-streaming result
        non_stream_result = llm.generate_single(prompt, sampling_params)
        non_stream_text = non_stream_result["generated_text"]
        
        # Get streaming result
        stream_tokens = []
        for result in llm.generate_stream(prompt, sampling_params):
            stream_tokens.append(result["token"])
            if result["finished"]:
                break
        
        stream_text = "".join(stream_tokens)
        
        # Results should be identical with temperature=0.0
        assert stream_text == non_stream_text, "Streaming and non-streaming should produce identical results"
    
    def test_streaming_with_different_sampling_params(self, llm):
        """Test streaming with different sampling parameters."""
        prompt = "Write a short story about a robot."
        
        # Test with different parameters
        test_cases = [
            SamplingParams(max_tokens=5, temperature=0.0),
            SamplingParams(max_tokens=10, temperature=0.7),
            SamplingParams(max_tokens=15, top_p=0.9),
            SamplingParams(max_tokens=8, top_k=50)
        ]
        
        for sampling_params in test_cases:
            tokens = []
            for result in llm.generate_stream(prompt, sampling_params):
                tokens.append(result["token"])
                if result["finished"]:
                    break
            
            # Should generate tokens up to max_tokens
            assert len(tokens) <= sampling_params.max_tokens, f"Should not exceed max_tokens {sampling_params.max_tokens}"
            assert len(tokens) > 0, "Should generate at least one token"
    
    def test_streaming_chat(self, llm):
        """Test streaming chat functionality."""
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
        
        # Test chat streaming
        tokens = []
        for result in llm.chat_stream(messages, sampling_params):
            tokens.append(result["token"])
            if result["finished"]:
                break
        
        assert len(tokens) > 0, "Chat streaming should generate tokens"
        
        # Verify the response mentions Paris
        response = "".join(tokens)
        assert "Paris" in response or "France" in response, "Should mention Paris or France"
    
    def test_streaming_error_handling(self, llm):
        """Test streaming error handling."""
        # Test with invalid prompt (empty string)
        prompt = ""
        sampling_params = SamplingParams(max_tokens=5)
        
        results = list(llm.generate_stream(prompt, sampling_params))
        
        # Should handle gracefully
        assert len(results) > 0, "Should return at least one result"
        assert results[-1]["finished"], "Should finish even with empty prompt"
    
    def test_streaming_performance(self, llm):
        """Test streaming performance characteristics."""
        prompt = "Explain the theory of relativity."
        sampling_params = SamplingParams(max_tokens=20, temperature=0.0)
        
        # Measure streaming performance
        start_time = time.time()
        token_count = 0
        
        for result in llm.generate_stream(prompt, sampling_params):
            token_count += 1
            if result["finished"]:
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert token_count > 0, "Should generate tokens"
        assert duration > 0, "Should take some time"
        assert duration < 30, "Should complete within reasonable time (30s)"
        
        # Calculate tokens per second
        tokens_per_sec = token_count / duration
        assert tokens_per_sec > 0.1, "Should generate at least 0.1 tokens per second"
    
    def test_streaming_metadata(self, llm):
        """Test that streaming provides correct metadata."""
        prompt = "Hello world"
        sampling_params = SamplingParams(max_tokens=5, temperature=0.0)
        
        request_id = None
        total_tokens = 0
        
        for result in llm.generate_stream(prompt, sampling_params):
            # Check request_id consistency
            if request_id is None:
                request_id = result["request_id"]
            else:
                assert result["request_id"] == request_id, "Request ID should be consistent"
            
            # Check token count progression
            assert result["tokens_generated"] == total_tokens + 1, "Token count should increment"
            total_tokens = result["tokens_generated"]
            
            if result["finished"]:
                break
        
        assert request_id is not None, "Should have a request ID"
        assert total_tokens > 0, "Should generate tokens"
    
    def test_streaming_stop_conditions(self, llm):
        """Test streaming with stop conditions."""
        prompt = "Count from 1 to 10:"
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
            stop_sequences=["10"]
        )
        
        tokens = []
        for result in llm.generate_stream(prompt, sampling_params):
            tokens.append(result["token"])
            if result["finished"]:
                break
        
        # Should stop when encountering "10"
        response = "".join(tokens)
        assert "10" in response, "Should include the stop sequence"
    
    def test_streaming_memory_cleanup(self, llm):
        """Test that streaming properly cleans up memory."""
        prompt = "Generate a long response about artificial intelligence."
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
        
        # Get initial memory stats
        initial_stats = llm.get_stats()
        initial_memory = initial_stats["memory_stats"]
        
        # Run streaming generation
        for result in llm.generate_stream(prompt, sampling_params):
            if result["finished"]:
                break
        
        # Get final memory stats
        final_stats = llm.get_stats()
        final_memory = final_stats["memory_stats"]
        
        # Memory should be properly managed (similar or better than initial)
        assert final_memory["allocated_blocks"] <= initial_memory["allocated_blocks"] + 10, "Memory should be properly managed"


if __name__ == "__main__":
    # Run basic streaming test
    llm = LLM(
        model_name="Qwen/Qwen3-0.6B",
        device="mps",
        max_queue_size=10,
        num_blocks=100,
        block_size=8
    )
    
    try:
        print("Testing streaming functionality...")
        prompt = "Hello, how are you today?"
        sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
        
        print(f"Prompt: {prompt}")
        print("Streaming response:")
        
        for result in llm.generate_stream(prompt, sampling_params):
            print(result["token"], end="", flush=True)
            if result["finished"]:
                break
        
        print("\n\nStreaming test completed successfully!")
        
    finally:
        llm.shutdown() 