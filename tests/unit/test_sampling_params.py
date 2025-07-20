"""
Tests for SamplingParams class.
"""

import pytest
from nano_qwen3_serving.core.sampling_params import SamplingParams


class TestSamplingParams:
    """Test cases for SamplingParams."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = SamplingParams()
        
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.max_tokens == 100
        assert params.min_tokens == 0
        assert params.do_sample is True
        assert params.use_beam_search is False
        assert params.use_cache is True
        assert params.cache_precision == "float16"
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            max_tokens=100,
            min_tokens=10
        )
        
        assert params.temperature == 0.8
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 100
        assert params.min_tokens == 10
    
    def test_preset_strategies(self):
        """Test preset sampling strategies."""
        # Test greedy preset
        greedy = SamplingParams.greedy(max_tokens=50)
        assert greedy.temperature == 0.0
        assert greedy.do_sample is False
        assert greedy.use_beam_search is False
        assert greedy.max_tokens == 50
        
        # Test creative preset
        creative = SamplingParams.creative(max_tokens=100)
        assert creative.temperature == 0.8
        assert creative.top_p == 0.9
        assert creative.top_k == 50
        assert creative.max_tokens == 100
        
        # Test balanced preset
        balanced = SamplingParams.balanced(max_tokens=75)
        assert balanced.temperature == 0.7
        assert balanced.top_p == 0.9
        assert balanced.top_k == 40
        assert balanced.max_tokens == 75
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test valid temperature range
        params = SamplingParams(temperature=0.5)
        assert params.temperature == 0.5
        
        # Test temperature at boundary
        params = SamplingParams(temperature=0.0)
        assert params.temperature == 0.0
        
        # Test invalid temperature (should raise error)
        with pytest.raises(ValueError):
            SamplingParams(temperature=-0.1)
        
        with pytest.raises(ValueError):
            SamplingParams(temperature=2.1)
        
        # Test valid top_p range
        params = SamplingParams(top_p=0.5)
        assert params.top_p == 0.5
        
        # Test invalid top_p
        with pytest.raises(ValueError):
            SamplingParams(top_p=1.1)
        
        with pytest.raises(ValueError):
            SamplingParams(top_p=-0.1)
        
        # Test valid top_k
        params = SamplingParams(top_k=10)
        assert params.top_k == 10
        
        # Test invalid top_k
        with pytest.raises(ValueError):
            SamplingParams(top_k=0)
    
    def test_token_limits_validation(self):
        """Test token limits validation."""
        # Test valid limits
        params = SamplingParams(min_tokens=10, max_tokens=100)
        assert params.min_tokens == 10
        assert params.max_tokens == 100
        
        # Test invalid limits (min > max)
        with pytest.raises(ValueError):
            SamplingParams(min_tokens=100, max_tokens=10)
    
    def test_sampling_strategy_validation(self):
        """Test sampling strategy validation."""
        # Test beam search with sampling (should warn but not error)
        params = SamplingParams(use_beam_search=True, do_sample=True)
        assert params.use_beam_search is True
        assert params.do_sample is True
    
    def test_to_dict(self):
        """Test to_dict method."""
        params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=100
        )
        
        params_dict = params.to_dict()
        
        assert isinstance(params_dict, dict)
        assert params_dict["temperature"] == 0.8
        assert params_dict["top_p"] == 0.9
        assert params_dict["max_tokens"] == 100
    
    def test_str_representation(self):
        """Test string representation."""
        # Test default params
        params = SamplingParams()
        assert str(params) == "SamplingParams(max_tokens=100)"
        
        # Test custom params
        params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=100)
        assert "temp=0.8" in str(params)
        assert "top_p=0.9" in str(params)
        assert "max_tokens=100" in str(params)
    
    def test_equality(self):
        """Test parameter equality."""
        params1 = SamplingParams(temperature=0.8, top_p=0.9)
        params2 = SamplingParams(temperature=0.8, top_p=0.9)
        params3 = SamplingParams(temperature=0.9, top_p=0.8)
        
        assert params1 == params2
        assert params1 != params3
    
    def test_stop_sequences(self):
        """Test stop sequences configuration."""
        stop_sequences = ["END", "STOP"]
        params = SamplingParams(stop_sequences=stop_sequences)
        
        assert params.stop_sequences == stop_sequences
    
    def test_stop_token_ids(self):
        """Test stop token IDs configuration."""
        stop_token_ids = [1, 2, 3]
        params = SamplingParams(stop_token_ids=stop_token_ids)
        
        assert params.stop_token_ids == stop_token_ids
    
    def test_repetition_penalty(self):
        """Test repetition penalty configuration."""
        params = SamplingParams(repetition_penalty=1.2)
        assert params.repetition_penalty == 1.2
        
        # Test invalid repetition penalty
        with pytest.raises(ValueError):
            SamplingParams(repetition_penalty=-0.1)
    
    def test_length_penalty(self):
        """Test length penalty configuration."""
        params = SamplingParams(length_penalty=0.8)
        assert params.length_penalty == 0.8
    
    def test_num_beams(self):
        """Test number of beams configuration."""
        params = SamplingParams(num_beams=4)
        assert params.num_beams == 4
        
        # Test invalid number of beams
        with pytest.raises(ValueError):
            SamplingParams(num_beams=0) 