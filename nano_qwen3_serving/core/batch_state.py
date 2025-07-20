"""
Structured batch state objects for continuous batching.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch


@dataclass
class SequenceInfo:
    """Information about a sequence in the current batch."""
    sequence_id: int
    request_id: int
    start_position: int  # Position in batch tensor
    current_length: int  # Current sequence length
    max_new_tokens: int  # Remaining tokens to generate
    sampling_params: Any  # SamplingParams object
    is_complete: bool = False
    completion_reason: Optional[str] = None


@dataclass
class BatchState:
    """Complete state of the current batch for engine processing."""
    # Core tensors
    input_ids: torch.Tensor  # (batch_size, seq_len)
    attention_mask: torch.Tensor  # (batch_size, seq_len)
    
    # Sequence mapping
    sequence_map: Dict[int, SequenceInfo]  # sequence_id -> SequenceInfo
    position_to_sequence: Dict[int, int]  # position -> sequence_id
    
    # Batch metadata
    batch_size: int
    max_seq_length: int
    active_sequences: List[int]  # List of active sequence_ids
    
    # Performance tracking
    step_count: int = 0
    total_tokens_processed: int = 0


@dataclass
class BatchUpdate:
    """Result of batch processing for scheduler update."""
    # New tokens generated for each sequence
    new_tokens: Dict[int, List[int]]  # sequence_id -> [token_ids]
    
    # Completed sequences
    completed_sequences: List[int]  # sequence_ids that finished
    
    # Model outputs (optional, for debugging)
    model_outputs: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    inference_time: float = 0.0
    tokens_generated: int = 0 