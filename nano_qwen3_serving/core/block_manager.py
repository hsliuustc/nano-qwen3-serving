"""
Memory management for GPU blocks and KV cache.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import psutil


@dataclass
class MemoryBlock:
    """Represents a memory block for storing KV cache."""
    start_idx: int
    end_idx: int
    is_allocated: bool = False
    sequence_id: Optional[int] = None
    last_used: float = 0.0


class BlockManager:
    """
    Manages GPU memory blocks for efficient KV cache storage.
    
    This class handles the allocation, deallocation, and reuse of memory blocks
    for storing key-value pairs in the attention mechanism.
    """
    
    def __init__(
        self,
        num_blocks: int = 1024,
        block_size: int = 16,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the block manager.
        
        Args:
            num_blocks: Number of memory blocks to allocate
            block_size: Size of each block in tokens
            device: Device to allocate memory on ("auto", "mps", "cuda", "cpu")
            dtype: Data type for the memory blocks (None for auto-detection)
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # Initialize device manager for device detection
        from .device_manager import DeviceManager
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        
        # Set dtype (auto-detect if not provided)
        if dtype is None:
            self.dtype = self.device_manager.get_dtype()
        else:
            self.dtype = dtype
        
        # Initialize memory blocks
        self.blocks: List[MemoryBlock] = []
        self._initialize_blocks()
        
        # Track allocated blocks
        self.allocated_blocks: Dict[int, List[int]] = {}  # sequence_id -> block_indices
        self.free_blocks: List[int] = list(range(num_blocks))
        
        # Memory statistics
        self.total_memory = num_blocks * block_size
        self.allocated_memory = 0
        
        logger.info(f"BlockManager initialized with {num_blocks} blocks of size {block_size}")
    
    def _initialize_blocks(self):
        """Initialize memory blocks."""
        for i in range(self.num_blocks):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            self.blocks.append(MemoryBlock(start_idx, end_idx))
    
    def allocate_blocks(self, sequence_id: int, num_tokens: int) -> List[int]:
        """
        Allocate blocks for a sequence.
        
        Args:
            sequence_id: Unique identifier for the sequence
            num_tokens: Number of tokens to allocate space for
            
        Returns:
            List of allocated block indices
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < num_blocks_needed:
            # Need to free some blocks
            self._evict_blocks(num_blocks_needed - len(self.free_blocks))
        
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(f"Insufficient memory blocks. Need {num_blocks_needed}, have {len(self.free_blocks)}")
        
        # Allocate blocks
        allocated_indices = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop()
            self.blocks[block_idx].is_allocated = True
            self.blocks[block_idx].sequence_id = sequence_id
            allocated_indices.append(block_idx)
        
        self.allocated_blocks[sequence_id] = allocated_indices
        self.allocated_memory += num_blocks_needed * self.block_size
        
        logger.debug(f"Allocated {num_blocks_needed} blocks for sequence {sequence_id}")
        return allocated_indices
    
    def free_sequence_blocks(self, sequence_id: int) -> None:
        """
        Free blocks allocated to a sequence.
        
        Args:
            sequence_id: ID of the sequence to free
        """
        if sequence_id not in self.allocated_blocks:
            return
        
        block_indices = self.allocated_blocks[sequence_id]
        for block_idx in block_indices:
            self.blocks[block_idx].is_allocated = False
            self.blocks[block_idx].sequence_id = None
            self.free_blocks.append(block_idx)
        
        self.allocated_memory -= len(block_indices) * self.block_size
        del self.allocated_blocks[sequence_id]
        
        logger.debug(f"Freed {len(block_indices)} blocks for sequence {sequence_id}")
    
    def _evict_blocks(self, num_blocks: int) -> None:
        """
        Evict blocks using LRU strategy.
        
        Args:
            num_blocks: Number of blocks to evict
        """
        # Simple LRU eviction - in practice, you might want more sophisticated strategies
        evicted = 0
        for sequence_id in list(self.allocated_blocks.keys()):
            if evicted >= num_blocks:
                break
            # Get the number of blocks before freeing
            num_blocks_in_sequence = len(self.allocated_blocks[sequence_id])
            self.free_sequence_blocks(sequence_id)
            evicted += num_blocks_in_sequence
        
        logger.debug(f"Evicted {evicted} blocks")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_blocks = self.num_blocks
        allocated_blocks = total_blocks - len(self.free_blocks)
        utilization = allocated_blocks / total_blocks if total_blocks > 0 else 0.0
        
        # Get device-specific memory stats
        device_stats = self.device_manager.get_memory_stats()
        
        return {
            "total_blocks": total_blocks,
            "allocated_blocks": allocated_blocks,
            "free_blocks": len(self.free_blocks),
            "utilization": utilization,
            "allocated_memory": self.allocated_memory,
            "total_memory": self.total_memory,
            "device_stats": device_stats
        }
    
    def create_kv_cache(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Create KV cache tensor for the allocated blocks.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            dtype: Data type for the cache (defaults to self.dtype)
            
        Returns:
            KV cache tensor of shape (num_layers, 2, num_blocks, num_heads, block_size, head_dim)
        """
        if dtype is None:
            dtype = self.dtype
        
        cache_shape = (num_layers, 2, self.num_blocks, num_heads, self.block_size, head_dim)
        cache = torch.zeros(cache_shape, dtype=dtype, device=self.device_manager.get_device())
        
        logger.info(f"Created KV cache with shape {cache_shape}")
        return cache
    
    def get_block_indices(self, sequence_id: int) -> List[int]:
        """Get block indices for a sequence."""
        return self.allocated_blocks.get(sequence_id, [])
    
    def is_sequence_allocated(self, sequence_id: int) -> bool:
        """Check if a sequence has allocated blocks."""
        return sequence_id in self.allocated_blocks
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / (1024**3),  # GB
            "available": memory.available / (1024**3),  # GB
            "used": memory.used / (1024**3),  # GB
            "percent": memory.percent
        } 