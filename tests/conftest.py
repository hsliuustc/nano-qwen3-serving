"""
Pytest configuration and fixtures for nano Qwen3 serving engine tests.
"""

import pytest
import torch
from nano_qwen3_serving import LLM, SamplingParams
from nano_qwen3_serving.core.block_manager import BlockManager
from nano_qwen3_serving.core.scheduler import Scheduler, RequestPriority


@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing."""
    # Auto-detect best available device
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture(scope="session")
def dtype():
    """Get the data type to use for testing."""
    return torch.float16


@pytest.fixture
def sampling_params():
    """Create a basic SamplingParams instance for testing."""
    return SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=10
    )


@pytest.fixture
def greedy_params():
    """Create greedy SamplingParams for testing."""
    return SamplingParams.greedy(max_tokens=10)


@pytest.fixture
def creative_params():
    """Create creative SamplingParams for testing."""
    return SamplingParams.creative(max_tokens=10)


@pytest.fixture
def balanced_params():
    """Create balanced SamplingParams for testing."""
    return SamplingParams.balanced(max_tokens=10)


@pytest.fixture
def block_manager(device, dtype):
    """Create a BlockManager instance for testing."""
    return BlockManager(
        num_blocks=10,
        block_size=8,
        device=device,
        dtype=dtype
    )


@pytest.fixture
def scheduler():
    """Create a Scheduler instance for testing."""
    return Scheduler(max_queue_size=10)


@pytest.fixture
def llm_small(device, dtype):
    """Create a small LLM instance for testing (minimal settings)."""
    llm = LLM(
        model_name="Qwen/Qwen3-0.6B",
        device=device,
        dtype=str(dtype).split('.')[-1],  # Convert torch.float16 to "float16"
        max_queue_size=5,
        num_blocks=50,
        block_size=8,
        max_seq_length=1024
    )
    
    yield llm
    
    # Cleanup
    llm.shutdown()


@pytest.fixture
def llm_minimal(device, dtype):
    """Create a minimal LLM instance for quick tests."""
    llm = LLM(
        model_name="Qwen/Qwen3-0.6B",
        device=device,
        dtype=str(dtype).split('.')[-1],
        max_queue_size=3,
        num_blocks=20,
        block_size=4,
        max_seq_length=512
    )
    
    yield llm
    
    # Cleanup
    llm.shutdown()


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "mps: marks tests that require MPS"
    )


# Skip tests if MPS is not available
def pytest_collection_modifyitems(config, items):
    """Skip MPS tests if MPS is not available."""
    skip_mps = pytest.mark.skip(reason="MPS not available")
    
    for item in items:
        if "mps" in item.keywords and not torch.backends.mps.is_available():
            item.add_marker(skip_mps)


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_result_structure(result):
        """Assert that a generation result has the correct structure."""
        required_keys = [
            "request_id", "prompt", "generated_text", 
            "tokens_generated", "total_tokens", "block_indices"
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        assert isinstance(result["request_id"], int)
        assert isinstance(result["prompt"], str)
        assert isinstance(result["generated_text"], str)
        assert isinstance(result["tokens_generated"], int)
        assert isinstance(result["total_tokens"], int)
        assert isinstance(result["block_indices"], list)
    
    @staticmethod
    def assert_no_error(result):
        """Assert that a result has no error."""
        assert "error" not in result or result["error"] is None
    
    @staticmethod
    def assert_generation_success(result):
        """Assert that generation was successful."""
        TestUtils.assert_no_error(result)
        assert len(result["generated_text"]) > 0
        assert result["tokens_generated"] > 0
        assert result["total_tokens"] > 0


# Fixture for test data
@pytest.fixture
def test_prompts():
    """Common test prompts."""
    return [
        "Hello",
        "What is AI?",
        "Write a short story",
        "Explain quantum computing",
        "How are you today?"
    ]


@pytest.fixture
def test_messages():
    """Common test chat messages."""
    return [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "What is machine learning?"}
    ]


# Performance testing utilities
class PerformanceTest:
    """Utilities for performance testing."""
    
    @staticmethod
    def measure_generation_time(llm, prompt, sampling_params):
        """Measure the time taken for text generation."""
        import time
        
        start_time = time.time()
        result = llm.generate_single(prompt, sampling_params)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        return {
            "result": result,
            "generation_time": generation_time,
            "tokens_per_second": result["tokens_generated"] / generation_time if generation_time > 0 else 0
        }
    
    @staticmethod
    def benchmark_batch_processing(llm, prompts, sampling_params):
        """Benchmark batch processing performance."""
        import time
        
        start_time = time.time()
        results = llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = sum(r["tokens_generated"] for r in results if "error" not in r)
        
        return {
            "results": results,
            "total_time": total_time,
            "average_time_per_prompt": total_time / len(prompts),
            "total_tokens": total_tokens,
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0
        } 