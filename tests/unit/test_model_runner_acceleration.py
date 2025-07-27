"""
Tests for enhanced ModelRunner with acceleration features.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from nano_qwen3_serving.core.model_runner import ModelRunner
from nano_qwen3_serving.core.acceleration import AccelerationConfig, QuantizationType


def create_mock_model():
    """Create a properly mocked model instance for testing."""
    mock_model_instance = Mock()
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.num_parameters.return_value = 1000000
    
    # Mock parameters for memory calculation
    mock_param = Mock()
    mock_param.numel.return_value = 1000000
    mock_model_instance.parameters.return_value = [mock_param]
    
    # Mock named_modules for quantization info
    mock_model_instance.named_modules.return_value = [
        ("layer1", Mock(weight=Mock(dtype=torch.float32)))
    ]
    
    # Mock config for acceleration
    mock_model_instance.config = Mock()
    mock_model_instance.config.use_flash_attention_2 = False
    mock_model_instance.config.use_memory_efficient_attention = False
    mock_model_instance.config.use_fused_rms_norm = False
    mock_model_instance.config.use_fused_mlp = False
    
    # Mock gradient checkpointing
    mock_model_instance.gradient_checkpointing_enable = Mock()
    
    return mock_model_instance


class TestModelRunnerAcceleration:
    """Test ModelRunner acceleration features."""
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    def test_model_runner_with_acceleration_config(self, mock_tokenizer, mock_model):
        """Test ModelRunner initialization with acceleration config."""
        # Mock model and tokenizer
        mock_model_instance = create_mock_model()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create acceleration config
        acceleration_config = AccelerationConfig(
            quantization=QuantizationType.FP16,
            use_flash_attention=True,
            use_torch_compile=False
        )
        
        # Initialize ModelRunner
        runner = ModelRunner(
            model_name="test-model",
            device="cpu",
            acceleration_config=acceleration_config
        )
        
        assert runner.acceleration_config == acceleration_config
        assert runner.accelerator is not None
        assert "acceleration" in runner.get_model_info()
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    def test_model_runner_default_acceleration(self, mock_tokenizer, mock_model):
        """Test ModelRunner with default acceleration config."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.num_parameters.return_value = 1000000
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Initialize ModelRunner without acceleration config
        runner = ModelRunner(
            model_name="test-model",
            device="cpu"
        )
        
        assert runner.acceleration_config is not None
        assert runner.acceleration_config.quantization == QuantizationType.NONE
        assert runner.accelerator is not None
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    def test_get_acceleration_info(self, mock_tokenizer, mock_model):
        """Test getting acceleration information."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.num_parameters.return_value = 1000000
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        acceleration_config = AccelerationConfig(
            quantization=QuantizationType.DYNAMIC_INT8,
            use_flash_attention=True
        )
        
        runner = ModelRunner(
            model_name="test-model",
            device="cpu",
            acceleration_config=acceleration_config
        )
        
        info = runner.get_acceleration_info()
        
        assert "acceleration_config" in info
        assert "applied_optimizations" in info
        assert "memory_reductions" in info
        assert "performance_estimates" in info
        assert info["acceleration_config"]["quantization"] == "dynamic_int8"
        assert info["acceleration_config"]["use_flash_attention"] is True
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    def test_get_quantization_info(self, mock_tokenizer, mock_model):
        """Test getting quantization information."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.num_parameters.return_value = 1000000
        mock_model_instance.named_modules.return_value = [
            ("layer1", Mock(weight=Mock(dtype=torch.float32))),
            ("layer2", Mock(weight=Mock(dtype=torch.qint8)))
        ]
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        acceleration_config = AccelerationConfig(
            quantization=QuantizationType.DYNAMIC_INT8
        )
        
        runner = ModelRunner(
            model_name="test-model",
            device="cpu",
            acceleration_config=acceleration_config
        )
        
        info = runner.get_quantization_info()
        
        assert "quantization_type" in info
        assert "is_quantized" in info
        assert "supported_types" in info
        assert "quantized_layers" in info
        assert "num_quantized_layers" in info
        
        assert info["quantization_type"] == "dynamic_int8"
        assert info["is_quantized"] is True
        assert info["num_quantized_layers"] == 1  # Only layer2 has qint8
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    def test_estimate_memory_usage(self, mock_tokenizer, mock_model):
        """Test memory usage estimation."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.num_parameters.return_value = 1000000
        
        # Mock parameters for memory calculation
        mock_params = [Mock(numel=Mock(return_value=500000)), Mock(numel=Mock(return_value=500000))]
        mock_model_instance.parameters.return_value = mock_params
        
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        acceleration_config = AccelerationConfig(
            quantization=QuantizationType.FP16
        )
        
        runner = ModelRunner(
            model_name="test-model",
            device="cpu",
            dtype=torch.float32,
            acceleration_config=acceleration_config
        )
        
        memory_info = runner.estimate_memory_usage()
        
        assert "base_model_size_mb" in memory_info
        assert "accelerated_model_size_mb" in memory_info
        assert "memory_saved_mb" in memory_info
        assert "memory_reduction_percent" in memory_info
        assert "parameter_count" in memory_info
        assert "dtype" in memory_info
        assert "applied_optimizations" in memory_info
        
        assert memory_info["parameter_count"] == 1000000
        assert memory_info["base_model_size_mb"] > 0
        assert memory_info["accelerated_model_size_mb"] < memory_info["base_model_size_mb"]
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    @patch('nano_qwen3_serving.core.acceleration.benchmark_acceleration')
    def test_benchmark_performance(self, mock_benchmark, mock_tokenizer, mock_model):
        """Test performance benchmarking."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.num_parameters.return_value = 1000000
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock benchmark results
        mock_benchmark_results = {
            "avg_inference_time": 0.1,
            "min_inference_time": 0.08,
            "max_inference_time": 0.12,
            "total_time": 1.0,
            "iterations": 10,
            "tokens_per_second": 50.0
        }
        mock_benchmark.return_value = mock_benchmark_results
        
        runner = ModelRunner(
            model_name="test-model",
            device="cpu"
        )
        
        results = runner.benchmark_performance(
            sample_text="Hello world",
            num_iterations=10
        )
        
        # Check benchmark was called
        mock_benchmark.assert_called_once()
        
        # Check results contain expected fields
        assert "avg_inference_time" in results
        assert "applied_optimizations" in results
        assert "memory_reductions" in results
        assert "performance_estimates" in results
        assert "device" in results
        assert "dtype" in results
        assert "quantization" in results
        assert "sample_text" in results
        assert "input_length" in results
        
        assert results["sample_text"] == "Hello world"
        assert results["iterations"] == 10
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    def test_update_acceleration_config(self, mock_tokenizer, mock_model):
        """Test updating acceleration configuration."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.num_parameters.return_value = 1000000
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Initial config
        initial_config = AccelerationConfig(
            quantization=QuantizationType.NONE,
            use_flash_attention=False
        )
        
        runner = ModelRunner(
            model_name="test-model",
            device="cpu",
            acceleration_config=initial_config
        )
        
        # New config
        new_config = AccelerationConfig(
            quantization=QuantizationType.FP16,
            use_flash_attention=True
        )
        
        runner.update_acceleration_config(new_config)
        
        assert runner.acceleration_config == new_config
        assert runner.acceleration_config.quantization == QuantizationType.FP16
        assert runner.acceleration_config.use_flash_attention is True
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    def test_update_acceleration_config_failure_rollback(self, mock_tokenizer, mock_model):
        """Test rollback on acceleration config update failure."""
        # Mock model and tokenizer - first call succeeds, second fails
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.num_parameters.return_value = 1000000
        
        mock_model.from_pretrained.side_effect = [mock_model_instance, Exception("Load failed")]
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Initial config
        initial_config = AccelerationConfig(
            quantization=QuantizationType.NONE
        )
        
        runner = ModelRunner(
            model_name="test-model",
            device="cpu",
            acceleration_config=initial_config
        )
        
        # New config that will fail to load
        new_config = AccelerationConfig(
            quantization=QuantizationType.FP16
        )
        
        with pytest.raises(Exception, match="Load failed"):
            runner.update_acceleration_config(new_config)
        
        # Should rollback to original config
        assert runner.acceleration_config == initial_config
    
    @patch('nano_qwen3_serving.core.model_runner.AutoModelForCausalLM')
    @patch('nano_qwen3_serving.core.model_runner.AutoTokenizer')
    def test_enhanced_get_model_info(self, mock_tokenizer, mock_model):
        """Test enhanced model info with acceleration details."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.num_parameters.return_value = 1000000
        mock_model_instance.parameters.return_value = [Mock(numel=Mock(return_value=1000000))]
        mock_model_instance.named_modules.return_value = []
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        acceleration_config = AccelerationConfig(
            quantization=QuantizationType.DYNAMIC_INT8,
            use_flash_attention=True
        )
        
        runner = ModelRunner(
            model_name="test-model",
            device="cpu",
            acceleration_config=acceleration_config
        )
        
        info = runner.get_model_info()
        
        # Check basic model info
        assert "model_name" in info
        assert "device" in info
        assert "dtype" in info
        assert "num_parameters" in info
        
        # Check acceleration info
        assert "acceleration" in info
        assert "memory_usage" in info
        assert "quantization" in info
        
        # Check acceleration details
        assert info["acceleration"]["acceleration_config"]["quantization"] == "dynamic_int8"
        assert info["quantization"]["quantization_type"] == "dynamic_int8"
        assert info["memory_usage"]["parameter_count"] == 1000000


if __name__ == "__main__":
    # Run tests manually if needed
    import sys
    pytest.main([__file__] + sys.argv[1:])