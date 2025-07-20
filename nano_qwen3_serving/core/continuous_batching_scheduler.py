"""
Continuous batching scheduler with structured batch state management.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from .batch_state import BatchState, BatchUpdate, SequenceInfo
from .scheduler import RequestPriority


@dataclass
class Request:
    """Represents a generation request for continuous batching."""
    request_id: int
    prompt: str
    sampling_params: Any  # SamplingParams
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class ContinuousBatchingScheduler:
    """
    Scheduler that manages continuous batching with structured batch states.
    
    This scheduler maintains a dynamic batch of sequences and provides
    structured batch state objects for the engine to process.
    """
    
    def __init__(
        self, 
        max_queue_size: int = 1000, 
        max_batch_size: int = 8,
        max_wait_time: float = 0.01
    ):
        """
        Initialize the continuous batching scheduler.
        
        Args:
            max_queue_size: Maximum number of requests in queue
            max_batch_size: Maximum number of sequences in a batch
            max_wait_time: Maximum time to wait for batching (seconds)
        """
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        # Request management
        self.pending_requests: List[Request] = []
        self.active_sequences: Dict[int, SequenceInfo] = {}
        self.completed_results: Dict[int, Dict[str, Any]] = {}
        
        # Batch state
        self.current_batch_state: Optional[BatchState] = None
        self.next_sequence_id = 0
        
        # Statistics
        self.total_requests = 0
        self.completed_requests_count = 0
        self.failed_requests_count = 0
        
        logger.info(f"ContinuousBatchingScheduler initialized with max_batch_size={max_batch_size}")
    
    def add_request(self, request: Request) -> int:
        """
        Add new request to pending queue.
        
        Args:
            request: Request to add
            
        Returns:
            Request ID
        """
        if len(self.pending_requests) >= self.max_queue_size:
            raise RuntimeError("Request queue is full")
        
        self.pending_requests.append(request)
        self.total_requests += 1
        
        logger.debug(f"Added request {request.request_id} to pending queue")
        return request.request_id
    
    def get_batch_state(self) -> Optional[BatchState]:
        """
        Get current batch state for engine processing.
        
        Returns:
            BatchState if there are active sequences, None otherwise
        """
        if not self.active_sequences:
            return None
        
        return self.current_batch_state
    
    def update_batch(self, batch_update: BatchUpdate) -> None:
        """
        Update batch state after engine processing.
        
        Args:
            batch_update: Results from engine processing
        """
        # Update sequences with new tokens
        for sequence_id, new_tokens in batch_update.new_tokens.items():
            if sequence_id in self.active_sequences:
                seq_info = self.active_sequences[sequence_id]
                seq_info.current_length += len(new_tokens)
                
                # Check if sequence is complete
                if (len(new_tokens) == 0 or 
                    seq_info.current_length >= seq_info.max_new_tokens):
                    seq_info.is_complete = True
                    seq_info.completion_reason = "max_tokens_reached"
        
        # Handle completed sequences
        for sequence_id in batch_update.completed_sequences:
            self._complete_sequence(sequence_id)
        
        # Try to add new sequences from pending queue
        self._add_pending_requests_to_batch()
        
        # Update batch state
        self._update_batch_tensors()
        
        logger.debug(f"Updated batch: {len(self.active_sequences)} active sequences")
    
    def get_completed_results(self) -> List[Dict[str, Any]]:
        """
        Get results for completed sequences.
        
        Returns:
            List of completed results
        """
        results = list(self.completed_results.values())
        self.completed_results.clear()
        return results
    
    def _add_pending_requests_to_batch(self) -> None:
        """Add pending requests to current batch if possible."""
        current_time = time.time()
        
        # Remove expired requests
        self.pending_requests = [
            req for req in self.pending_requests
            if current_time - req.created_at < 30.0  # 30 second timeout
        ]
        
        while (self.pending_requests and 
               len(self.active_sequences) < self.max_batch_size):
            
            request = self.pending_requests.pop(0)
            sequence_id = self._get_next_sequence_id()
            
            # Create sequence info
            seq_info = SequenceInfo(
                sequence_id=sequence_id,
                request_id=request.request_id,
                start_position=len(self.active_sequences),
                current_length=0,
                max_new_tokens=request.sampling_params.max_tokens,
                sampling_params=request.sampling_params
            )
            
            self.active_sequences[sequence_id] = seq_info
            
            logger.debug(f"Added sequence {sequence_id} to batch")
        
        # Update batch state
        self._update_batch_tensors()
    
    def _update_batch_tensors(self) -> None:
        """Update batch tensors based on current sequences."""
        if not self.active_sequences:
            self.current_batch_state = None
            return
        
        # For now, we'll create a simple batch state
        # In a full implementation, this would handle tokenization and padding
        batch_size = len(self.active_sequences)
        
        # Create dummy tensors (placeholder for now)
        # In real implementation, these would be actual tokenized inputs
        dummy_input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
        dummy_attention_mask = torch.ones((batch_size, 1), dtype=torch.long)
        
        # Create position mappings
        position_to_sequence = {}
        for i, seq_info in enumerate(self.active_sequences.values()):
            position_to_sequence[i] = seq_info.sequence_id
        
        # Update batch state
        self.current_batch_state = BatchState(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            sequence_map=self.active_sequences.copy(),
            position_to_sequence=position_to_sequence,
            batch_size=batch_size,
            max_seq_length=1,  # Placeholder
            active_sequences=list(self.active_sequences.keys())
        )
    
    def _complete_sequence(self, sequence_id: int) -> None:
        """Mark a sequence as completed and store results."""
        if sequence_id not in self.active_sequences:
            return
        
        seq_info = self.active_sequences[sequence_id]
        
        # Create result
        result = {
            "request_id": seq_info.request_id,
            "sequence_id": sequence_id,
            "generated_text": "",  # Placeholder - would be actual generated text
            "tokens_generated": seq_info.current_length,
            "completion_reason": seq_info.completion_reason,
            "prompt": ""  # Placeholder - would be actual prompt
        }
        
        self.completed_results[sequence_id] = result
        del self.active_sequences[sequence_id]
        self.completed_requests_count += 1
        
        logger.debug(f"Completed sequence {sequence_id}")
    
    def _get_next_sequence_id(self) -> int:
        """Get next available sequence ID."""
        sequence_id = self.next_sequence_id
        self.next_sequence_id += 1
        return sequence_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "pending_requests": len(self.pending_requests),
            "active_sequences": len(self.active_sequences),
            "completed_requests": self.completed_requests_count,
            "failed_requests": self.failed_requests_count,
            "total_requests": self.total_requests,
            "batch_size": len(self.active_sequences) if self.active_sequences else 0,
            "has_batch": self.current_batch_state is not None
        }
    
    def clear_completed_results(self) -> None:
        """Clear completed results to free memory."""
        self.completed_results.clear()
    
    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        logger.info("Shutting down ContinuousBatchingScheduler")
        self.pending_requests.clear()
        self.active_sequences.clear()
        self.completed_results.clear()
        self.current_batch_state = None 