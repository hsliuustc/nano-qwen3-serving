"""
Tests for BlockManager class.
"""

import pytest
import torch
from nano_qwen3_serving.core.block_manager import BlockManager, MemoryBlock


class TestMemoryBlock:
    """Test cases for MemoryBlock."""
    
    def test_memory_block_creation(self):
        """Test MemoryBlock creation."""
        block = MemoryBlock(start_idx=0, end_idx=16)
        
        assert block.start_idx == 0
        assert block.end_idx == 16
        assert block.is_allocated is False
        assert block.sequence_id is None
        assert block.last_used == 0.0
    
    def test_memory_block_with_optional_params(self):
        """Test MemoryBlock with optional parameters."""
        block = MemoryBlock(
            start_idx=8,
            end_idx=24,
            is_allocated=True,
            sequence_id=123,
            last_used=1.5
        )
        
        assert block.start_idx == 8
        assert block.end_idx == 24
        assert block.is_allocated is True
        assert block.sequence_id == 123
        assert block.last_used == 1.5


class TestBlockManager:
    """Test cases for BlockManager."""
    
    def test_initialization(self):
        """Test BlockManager initialization."""
        manager = BlockManager(
            num_blocks=10,
            block_size=8,
            device="mps",
            dtype=torch.float16
        )
        
        assert manager.num_blocks == 10
        assert manager.block_size == 8
        assert manager.device == "mps"
        assert manager.dtype == torch.float16
        assert len(manager.blocks) == 10
        assert len(manager.free_blocks) == 10
        assert manager.total_memory == 80  # 10 * 8
        assert manager.allocated_memory == 0
    
    def test_block_initialization(self):
        """Test that blocks are properly initialized."""
        manager = BlockManager(num_blocks=5, block_size=4)
        
        for i, block in enumerate(manager.blocks):
            expected_start = i * 4
            expected_end = expected_start + 4
            assert block.start_idx == expected_start
            assert block.end_idx == expected_end
            assert block.is_allocated is False
            assert block.sequence_id is None
    
    def test_allocate_blocks(self):
        """Test block allocation."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        # Allocate blocks for a sequence
        sequence_id = 123
        num_tokens = 20  # Needs 3 blocks (20 / 8 = 2.5, rounded up to 3)
        
        allocated_indices = manager.allocate_blocks(sequence_id, num_tokens)
        
        assert len(allocated_indices) == 3
        assert len(manager.free_blocks) == 7
        assert manager.allocated_memory == 24  # 3 * 8
        assert sequence_id in manager.allocated_blocks
        
        # Check that blocks are marked as allocated
        for block_idx in allocated_indices:
            block = manager.blocks[block_idx]
            assert block.is_allocated is True
            assert block.sequence_id == sequence_id
    
    def test_free_blocks(self):
        """Test block deallocation."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        # Allocate blocks first
        sequence_id = 123
        allocated_indices = manager.allocate_blocks(sequence_id, 20)
        initial_free_count = len(manager.free_blocks)
        
        # Free the blocks
        manager.free_sequence_blocks(sequence_id)
        
        assert len(manager.free_blocks) == 10  # All blocks should be free
        assert sequence_id not in manager.allocated_blocks
        assert manager.allocated_memory == 0
        
        # Check that blocks are marked as free
        for block_idx in allocated_indices:
            block = manager.blocks[block_idx]
            assert block.is_allocated is False
            assert block.sequence_id is None
    
    def test_free_nonexistent_sequence(self):
        """Test freeing blocks for a sequence that doesn't exist."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        # Try to free blocks for a sequence that was never allocated
        manager.free_sequence_blocks(999)
        
        # Should not raise an error and should not change anything
        assert len(manager.free_blocks) == 10
        assert manager.allocated_memory == 0
    
    def test_insufficient_blocks(self):
        """Test behavior when there are insufficient blocks."""
        manager = BlockManager(num_blocks=1, block_size=8)
        
        # Allocate the only block
        manager.allocate_blocks(1, 8)  # Uses 1 block
        
        # Try to allocate more blocks than available (even after eviction)
        with pytest.raises(RuntimeError):
            manager.allocate_blocks(2, 16)  # Needs 2 blocks, but only 1 available
    
    def test_memory_stats(self):
        """Test memory statistics."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        # Get initial stats
        stats = manager.get_memory_stats()
        assert stats["total_blocks"] == 10
        assert stats["allocated_blocks"] == 0
        assert stats["free_blocks"] == 10
        assert stats["utilization"] == 0.0
        assert stats["allocated_memory"] == 0
        assert stats["total_memory"] == 80
        
        # Allocate some blocks
        manager.allocate_blocks(1, 20)  # Uses 3 blocks
        
        # Get updated stats
        stats = manager.get_memory_stats()
        assert stats["allocated_blocks"] == 3
        assert stats["free_blocks"] == 7
        assert stats["utilization"] == 0.3
        assert stats["allocated_memory"] == 24
    
    def test_create_kv_cache(self):
        """Test KV cache creation."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        cache = manager.create_kv_cache(
            num_layers=6,
            num_heads=8,
            head_dim=64
        )
        
        # Check cache shape: (num_layers, 2, num_blocks, num_heads, block_size, head_dim)
        expected_shape = (6, 2, 10, 8, 8, 64)
        assert cache.shape == expected_shape
        assert cache.dtype == torch.float16
        assert cache.device.type == "mps"
    
    def test_get_block_indices(self):
        """Test getting block indices for a sequence."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        # Allocate blocks
        sequence_id = 123
        allocated_indices = manager.allocate_blocks(sequence_id, 20)
        
        # Get block indices
        retrieved_indices = manager.get_block_indices(sequence_id)
        assert retrieved_indices == allocated_indices
        
        # Test for non-existent sequence
        empty_indices = manager.get_block_indices(999)
        assert empty_indices == []
    
    def test_is_sequence_allocated(self):
        """Test checking if a sequence has allocated blocks."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        # Initially no sequences are allocated
        assert not manager.is_sequence_allocated(123)
        
        # Allocate blocks
        manager.allocate_blocks(123, 20)
        assert manager.is_sequence_allocated(123)
        
        # Free blocks
        manager.free_sequence_blocks(123)
        assert not manager.is_sequence_allocated(123)
    
    def test_system_memory_info(self):
        """Test system memory information retrieval."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        memory_info = manager.get_system_memory_info()
        
        assert "total" in memory_info
        assert "available" in memory_info
        assert "used" in memory_info
        assert "percent" in memory_info
        
        # Values should be positive
        assert memory_info["total"] > 0
        assert memory_info["available"] >= 0
        assert memory_info["used"] >= 0
        assert 0 <= memory_info["percent"] <= 100
    
    def test_block_size_calculation(self):
        """Test that the correct number of blocks is calculated."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        # Test different token counts
        test_cases = [
            (1, 1),    # 1 token needs 1 block
            (8, 1),    # 8 tokens need 1 block
            (9, 2),    # 9 tokens need 2 blocks
            (16, 2),   # 16 tokens need 2 blocks
            (17, 3),   # 17 tokens need 3 blocks
        ]
        
        for num_tokens, expected_blocks in test_cases:
            sequence_id = num_tokens  # Use token count as sequence ID
            allocated_indices = manager.allocate_blocks(sequence_id, num_tokens)
            assert len(allocated_indices) == expected_blocks
            
            # Clean up
            manager.free_sequence_blocks(sequence_id)
    
    def test_concurrent_allocations(self):
        """Test multiple concurrent allocations."""
        manager = BlockManager(num_blocks=10, block_size=8)
        
        # Allocate blocks for multiple sequences
        allocations = [
            (1, 16),  # 2 blocks
            (2, 24),  # 3 blocks
            (3, 8),   # 1 block
        ]
        
        for sequence_id, num_tokens in allocations:
            allocated_indices = manager.allocate_blocks(sequence_id, num_tokens)
            assert len(allocated_indices) > 0
            assert manager.is_sequence_allocated(sequence_id)
        
        # Check total allocation
        assert manager.allocated_memory == 48  # 2+3+1 = 6 blocks * 8 = 48
        assert len(manager.free_blocks) == 4   # 10 - 6 = 4 blocks remaining
        
        # Clean up
        for sequence_id, _ in allocations:
            manager.free_sequence_blocks(sequence_id)
        
        assert manager.allocated_memory == 0
        assert len(manager.free_blocks) == 10 