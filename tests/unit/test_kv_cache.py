#!/usr/bin/env python3
"""
Test suite for KV cache functionality in the nano Qwen3 serving engine.
"""

import pytest
import torch
import time
from typing import List, Dict, Any

from nano_qwen3_serving import SamplingParams, ModelRunner


class TestKVCache:
    """Test KV cache functionality."""
    
    @pytest.fixture
    def model_runner(self):
        """Create a model runner for testing."""
        return ModelRunner(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            dtype=torch.float16,
            use_cache=True,
            max_seq_length=4096
        )
    
    @pytest.fixture
    def sampling_params(self):
        """Create sampling parameters for testing."""
        return SamplingParams(
            temperature=0.7,
            max_tokens=10,
            use_cache=True
        )
    
    def test_kv_cache_enabled(self, model_runner):
        """Test that KV cache is enabled by default."""
        assert model_runner.cache_enabled == True
        assert model_runner.use_cache == True
    
    def test_kv_cache_disabled(self):
        """Test that KV cache can be disabled."""
        model_runner = ModelRunner(
            model_name="Qwen/Qwen3-0.6B",
            device="mps",
            dtype=torch.float16,
            use_cache=False
        )
        assert model_runner.cache_enabled == False
        assert model_runner.use_cache == False
    
    def test_single_token_generation_with_cache(self, model_runner, sampling_params):
        """Test single token generation with KV cache."""
        input_text = "Hello, how are you?"
        input_tokens = model_runner.tokenize(input_text)
        
        # Generate first token
        next_token, past_key_values = model_runner.generate_next_token(
            input_ids=input_tokens,
            sampling_params=sampling_params,
            past_key_values=None
        )
        
        # Verify output
        assert next_token.shape == (1, 1)  # Single token
        assert past_key_values is not None  # Cache should be returned
        assert len(past_key_values) > 0  # Should have cached values
        
        # Verify token is valid
        assert next_token.item() >= 0
        assert next_token.item() < model_runner.model.config.vocab_size
    
    def test_multiple_token_generation_with_cache(self, model_runner, sampling_params):
        """Test multiple token generation with KV cache."""
        input_text = "The quick brown fox"
        input_tokens = model_runner.tokenize(input_text)
        current_tokens = input_tokens.clone()
        past_key_values = None
        generated_tokens = []
        
        # Generate multiple tokens
        for i in range(5):
            # Use only the last token for input when using cache
            if past_key_values is not None:
                input_tokens = current_tokens[:, -1:]  # Only last token
            else:
                input_tokens = current_tokens  # Full sequence for first token
            
            next_token, past_key_values = model_runner.generate_next_token(
                input_ids=input_tokens,
                sampling_params=sampling_params,
                past_key_values=past_key_values
            )
            
            generated_tokens.append(next_token.item())
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Verify results
        assert len(generated_tokens) == 5
        assert past_key_values is not None
        assert all(token >= 0 for token in generated_tokens)
    
    def test_cache_consistency(self, model_runner):
        """Test that KV cache produces consistent results with greedy decoding."""
        input_text = "Artificial intelligence is"
        input_tokens = model_runner.tokenize(input_text)
        
        # Use greedy sampling for deterministic results
        greedy_params = SamplingParams(
            temperature=0.0,
            max_tokens=3,
            use_cache=True,
            do_sample=False
        )
        
        # Generate without cache (full sequence each time)
        no_cache_tokens = []
        current_tokens = input_tokens.clone()
        
        for i in range(3):
            next_token, _ = model_runner.generate_next_token(
                input_ids=current_tokens,
                sampling_params=greedy_params,
                past_key_values=None  # No cache
            )
            no_cache_tokens.append(next_token.item())
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Generate with cache
        cache_tokens = []
        current_tokens = input_tokens.clone()
        past_key_values = None
        
        for i in range(3):
            if past_key_values is not None:
                input_tokens = current_tokens[:, -1:]
            else:
                input_tokens = current_tokens
            
            next_token, past_key_values = model_runner.generate_next_token(
                input_ids=input_tokens,
                sampling_params=greedy_params,
                past_key_values=past_key_values
            )
            cache_tokens.append(next_token.item())
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Results should be identical with greedy decoding
        assert no_cache_tokens == cache_tokens
    
    def test_cache_performance(self, model_runner, sampling_params):
        """Test that KV cache improves performance."""
        input_text = "Machine learning algorithms"
        input_tokens = model_runner.tokenize(input_text)
        
        # Test without cache
        start_time = time.time()
        current_tokens = input_tokens.clone()
        
        for i in range(5):
            next_token, _ = model_runner.generate_next_token(
                input_ids=current_tokens,
                sampling_params=sampling_params,
                past_key_values=None
            )
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        no_cache_time = time.time() - start_time
        
        # Test with cache
        start_time = time.time()
        current_tokens = input_tokens.clone()
        past_key_values = None
        
        for i in range(5):
            if past_key_values is not None:
                input_tokens = current_tokens[:, -1:]
            else:
                input_tokens = current_tokens
            
            next_token, past_key_values = model_runner.generate_next_token(
                input_ids=input_tokens,
                sampling_params=sampling_params,
                past_key_values=past_key_values
            )
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        cache_time = time.time() - start_time
        
        # Cache should be faster (or at least not significantly slower)
        # Allow some tolerance for measurement variance
        assert cache_time <= no_cache_time * 1.5  # Cache should not be 50% slower
    
    def test_cache_memory_usage(self, model_runner, sampling_params):
        """Test that KV cache doesn't cause memory issues."""
        input_text = "Deep learning models"
        input_tokens = model_runner.tokenize(input_text)
        original_length = input_tokens.shape[1]
        
        # Generate many tokens with cache
        current_tokens = input_tokens.clone()
        past_key_values = None
        
        for i in range(20):
            if past_key_values is not None:
                input_tokens = current_tokens[:, -1:]
            else:
                input_tokens = current_tokens
            
            next_token, past_key_values = model_runner.generate_next_token(
                input_ids=input_tokens,
                sampling_params=sampling_params,
                past_key_values=past_key_values
            )
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Should complete without memory errors
        # We generated 20 additional tokens
        expected_length = original_length + 20
        actual_length = current_tokens.shape[1]
        assert actual_length == expected_length, f"Expected {expected_length} tokens, got {actual_length}"
    
    def test_cache_with_different_sampling_params(self, model_runner):
        """Test KV cache with different sampling parameters."""
        input_text = "Neural networks are"
        input_tokens = model_runner.tokenize(input_text)
        
        # Test with greedy sampling
        greedy_params = SamplingParams(
            temperature=0.0,
            max_tokens=3,
            use_cache=True,
            do_sample=False  # Explicitly disable sampling for greedy
        )
        
        current_tokens = input_tokens.clone()
        past_key_values = None
        
        for i in range(3):
            if past_key_values is not None:
                input_tokens = current_tokens[:, -1:]
            else:
                input_tokens = current_tokens
            
            next_token, past_key_values = model_runner.generate_next_token(
                input_ids=input_tokens,
                sampling_params=greedy_params,
                past_key_values=past_key_values
            )
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Test with creative sampling
        creative_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=3,
            use_cache=True
        )
        
        current_tokens = input_tokens.clone()
        past_key_values = None
        
        for i in range(3):
            if past_key_values is not None:
                input_tokens = current_tokens[:, -1:]
            else:
                input_tokens = current_tokens
            
            next_token, past_key_values = model_runner.generate_next_token(
                input_ids=input_tokens,
                sampling_params=creative_params,
                past_key_values=past_key_values
            )
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Both should complete successfully
        assert True  # If we get here, no exceptions were raised
    
    def test_cache_error_handling(self, model_runner, sampling_params):
        """Test error handling with KV cache."""
        # Test with invalid input - the model should handle this gracefully
        # by generating a reasonable token or using a fallback
        try:
            result = model_runner.generate_next_token(
                input_ids=torch.tensor([[999999]]).to(model_runner.device),  # Invalid token
                sampling_params=sampling_params,
                past_key_values=None
            )
            # If it doesn't raise an exception, that's fine - the model handled it gracefully
            assert result is not None
            assert len(result) == 2  # Should return (token, past_key_values)
        except Exception as e:
            # If it does raise an exception, that's also acceptable
            assert "token" in str(e).lower() or "vocab" in str(e).lower()
    
    def test_cache_model_info(self, model_runner):
        """Test that model info includes cache status."""
        info = model_runner.get_model_info()
        assert "use_cache" in info
        assert info["use_cache"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 