"""
Tests for model acceleration functionality.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from nano_qwen3_serving.core.acceleration import (
    AccelerationConfig,
    ModelAccelerator,
    QuantizationType,
    create_acceleration_config,
    benchmark_acceleration
)


class SimpleTestModel(nn.Module):
    """Simple model for testing acceleration."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.config = Mock()
        self.config.use_flash_attention_2 = False
        self.config.use_memory_efficient_attention = False
        self.config.use_fused_rms_norm = False
        self.config.use_fused_mlp = False
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class TestAccelerationConfig:
    """Test AccelerationConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AccelerationConfig()
        assert config.quantization == QuantizationType.NONE
        assert config.use_flash_attention is True
        assert config.use_torch_compile is False
        assert config.compile_mode == "reduce-overhead"
        assert config.use_channels_last is True
        assert config.use_gradient_checkpointing is True
        assert config.use_fused_kernels is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AccelerationConfig(
            quantization=QuantizationType.DYNAMIC_INT8,
            use_flash_attention=False,
            use_torch_compile=True,
            compile_mode="max-autotune"
        )
        assert config.quantization == QuantizationType.DYNAMIC_INT8
        assert config.use_flash_attention is False
        assert config.use_torch_compile is True
        assert config.compile_mode == "max-autotune"
    
    def test_create_acceleration_config(self):
        """Test creating config from string parameters."""
        config = create_acceleration_config(
            quantization="dynamic_int8",
            use_flash_attention=False,
            use_torch_compile=True
        )
        assert config.quantization == QuantizationType.DYNAMIC_INT8
        assert config.use_flash_attention is False
        assert config.use_torch_compile is True
    
    def test_invalid_quantization_string(self):
        """Test handling of invalid quantization string."""
        config = create_acceleration_config(quantization="invalid")
        assert config.quantization == QuantizationType.NONE


class TestModelAccelerator:
    """Test ModelAccelerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = SimpleTestModel()
        self.device = torch.device("cpu")
    
    def test_no_acceleration(self):
        """Test accelerator with no optimizations."""
        config = AccelerationConfig(
            quantization=QuantizationType.NONE,
            use_flash_attention=False,
            use_torch_compile=False,
            use_channels_last=False,
            use_gradient_checkpointing=False,
            use_fused_kernels=False
        )
        accelerator = ModelAccelerator(config)
        
        original_model = self.model
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        
        # Model should be unchanged
        assert accelerated_model is not None
        assert len(accelerator.applied_optimizations) == 0
    
    def test_fp16_quantization(self):
        """Test FP16 quantization."""
        config = AccelerationConfig(
            quantization=QuantizationType.FP16,
            use_flash_attention=False,
            use_torch_compile=False,
            use_channels_last=False,
            use_gradient_checkpointing=False,
            use_fused_kernels=False
        )
        accelerator = ModelAccelerator(config)
        
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        
        assert "fp16_quantization" in accelerator.applied_optimizations
        # Check if model parameters are in half precision
        for param in accelerated_model.parameters():
            assert param.dtype == torch.float16
    
    def test_bf16_quantization(self):
        """Test BF16 quantization."""
        config = AccelerationConfig(
            quantization=QuantizationType.BF16,
            use_flash_attention=False,
            use_torch_compile=False,
            use_channels_last=False,
            use_gradient_checkpointing=False,
            use_fused_kernels=False
        )
        accelerator = ModelAccelerator(config)
        
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        
        assert "bf16_quantization" in accelerator.applied_optimizations
        # Check if model parameters are in bfloat16 precision
        for param in accelerated_model.parameters():
            assert param.dtype == torch.bfloat16
    
    def test_dynamic_int8_quantization(self):
        """Test dynamic INT8 quantization."""
        config = AccelerationConfig(
            quantization=QuantizationType.DYNAMIC_INT8,
            use_flash_attention=False,
            use_torch_compile=False,
            use_channels_last=False,
            use_gradient_checkpointing=False,
            use_fused_kernels=False
        )
        accelerator = ModelAccelerator(config)
        
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        
        assert "dynamic_int8_quantization" in accelerator.applied_optimizations
        # Model should be quantized (hard to test exact quantization without complex setup)
        assert accelerated_model is not None
    
    def test_flash_attention_optimization(self):
        """Test flash attention optimization."""
        config = AccelerationConfig(
            quantization=QuantizationType.NONE,
            use_flash_attention=True,
            use_torch_compile=False,
            use_channels_last=False,
            use_gradient_checkpointing=False,
            use_fused_kernels=False
        )
        accelerator = ModelAccelerator(config)
        
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        
        # Should attempt to enable flash attention
        assert accelerated_model.config.use_flash_attention_2 is True
        assert accelerated_model.config.use_memory_efficient_attention is True
        assert "flash_attention_2" in accelerator.applied_optimizations
        assert "memory_efficient_attention" in accelerator.applied_optimizations
    
    def test_fused_kernels_optimization(self):
        """Test fused kernels optimization."""
        config = AccelerationConfig(
            quantization=QuantizationType.NONE,
            use_flash_attention=False,
            use_torch_compile=False,
            use_channels_last=False,
            use_gradient_checkpointing=False,
            use_fused_kernels=True
        )
        accelerator = ModelAccelerator(config)
        
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        
        # Should attempt to enable fused kernels
        assert accelerated_model.config.use_fused_rms_norm is True
        assert accelerated_model.config.use_fused_mlp is True
        assert "fused_rms_norm" in accelerator.applied_optimizations
        assert "fused_mlp" in accelerator.applied_optimizations
    
    @patch('torch.compile')
    def test_torch_compile_optimization(self, mock_compile):
        """Test torch.compile optimization."""
        mock_compile.return_value = self.model
        
        config = AccelerationConfig(
            quantization=QuantizationType.NONE,
            use_flash_attention=False,
            use_torch_compile=True,
            compile_mode="reduce-overhead",
            use_channels_last=False,
            use_gradient_checkpointing=False,
            use_fused_kernels=False
        )
        accelerator = ModelAccelerator(config)
        
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        
        mock_compile.assert_called_once_with(self.model, mode="reduce-overhead")
        assert "torch_compile_reduce-overhead" in accelerator.applied_optimizations
    
    def test_memory_footprint_reduction(self):
        """Test memory footprint reduction estimates."""
        config = AccelerationConfig(
            quantization=QuantizationType.DYNAMIC_INT8,
            use_gradient_checkpointing=True
        )
        accelerator = ModelAccelerator(config)
        
        # Add gradient_checkpointing_enable method to mock model
        self.model.gradient_checkpointing_enable = Mock()
        
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        reductions = accelerator.get_memory_footprint_reduction()
        
        assert "dynamic_int8_quantization" in reductions
        assert "gradient_checkpointing" in reductions
        assert reductions["dynamic_int8_quantization"] == 50.0
        assert reductions["gradient_checkpointing"] == 30.0
    
    def test_performance_estimates(self):
        """Test performance improvement estimates."""
        config = AccelerationConfig(
            quantization=QuantizationType.NONE,
            use_flash_attention=True,
            use_fused_kernels=True
        )
        accelerator = ModelAccelerator(config)
        
        accelerated_model = accelerator.apply_accelerations(self.model, self.device)
        speedups = accelerator.get_performance_estimates()
        
        assert "flash_attention_2" in speedups
        assert "memory_efficient_attention" in speedups
        assert "fused_rms_norm" in speedups
        assert "fused_mlp" in speedups
        assert speedups["flash_attention_2"] == 2.0
        assert speedups["memory_efficient_attention"] == 1.3


class TestBenchmarkAcceleration:
    """Test benchmark_acceleration function."""
    
    def test_benchmark_simple_model(self):
        """Test benchmarking with simple model."""
        model = SimpleTestModel()
        input_ids = torch.randn(1, 10)  # Use float input instead of int
        
        results = benchmark_acceleration(model, input_ids, num_iterations=3)
        
        assert "avg_inference_time" in results
        assert "min_inference_time" in results
        assert "max_inference_time" in results
        assert "total_time" in results
        assert "iterations" in results
        assert "tokens_per_second" in results
        
        assert results["iterations"] == 3
        assert results["avg_inference_time"] > 0
        assert results["min_inference_time"] <= results["avg_inference_time"]
        assert results["max_inference_time"] >= results["avg_inference_time"]
        assert results["tokens_per_second"] > 0
    
    def test_benchmark_error_handling(self):
        """Test benchmark error handling."""
        # Create a model that will fail
        model = Mock()
        model.side_effect = RuntimeError("Test error")
        input_ids = torch.randn(1, 10)  # Use float input
        
        with pytest.raises(RuntimeError):
            benchmark_acceleration(model, input_ids, num_iterations=1)


class TestQuantizationType:
    """Test QuantizationType enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert QuantizationType.NONE.value == "none"
        assert QuantizationType.DYNAMIC_INT8.value == "dynamic_int8"
        assert QuantizationType.STATIC_INT8.value == "static_int8"
        assert QuantizationType.INT4.value == "int4"
        assert QuantizationType.FP16.value == "fp16"
        assert QuantizationType.BF16.value == "bf16"
    
    def test_enum_iteration(self):
        """Test iterating over enum values."""
        values = [q.value for q in QuantizationType]
        expected = ["none", "dynamic_int8", "static_int8", "int4", "fp16", "bf16"]
        assert values == expected


if __name__ == "__main__":
    # Run tests manually if needed
    import sys
    pytest.main([__file__] + sys.argv[1:])