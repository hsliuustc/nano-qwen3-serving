"""
Tests for continuous batching functionality.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from nano_qwen3_serving.core.batch_state import BatchState, BatchUpdate, SequenceInfo
from nano_qwen3_serving.core.continuous_batching_scheduler import ContinuousBatchingScheduler, Request
from nano_qwen3_serving.core.sampling_params import SamplingParams


class TestBatchState:
    """Test batch state dataclasses."""
    
    def test_sequence_info_creation(self):
        """Test SequenceInfo creation."""
        sampling_params = SamplingParams(max_tokens=10)
        seq_info = SequenceInfo(
            sequence_id=1,
            request_id=100,
            start_position=0,
            current_length=5,
            max_new_tokens=10,
            sampling_params=sampling_params
        )
        
        assert seq_info.sequence_id == 1
        assert seq_info.request_id == 100
        assert seq_info.start_position == 0
        assert seq_info.current_length == 5
        assert seq_info.max_new_tokens == 10
        assert seq_info.is_complete == False
        assert seq_info.completion_reason is None
    
    def test_batch_state_creation(self):
        """Test BatchState creation."""
        input_ids = torch.zeros((2, 10), dtype=torch.long)
        attention_mask = torch.ones((2, 10), dtype=torch.long)
        
        sequence_map = {
            1: SequenceInfo(1, 100, 0, 5, 10, SamplingParams()),
            2: SequenceInfo(2, 101, 1, 3, 8, SamplingParams())
        }
        
        position_to_sequence = {0: 1, 1: 2}
        
        batch_state = BatchState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_map=sequence_map,
            position_to_sequence=position_to_sequence,
            batch_size=2,
            max_seq_length=10,
            active_sequences=[1, 2]
        )
        
        assert batch_state.batch_size == 2
        assert batch_state.max_seq_length == 10
        assert len(batch_state.active_sequences) == 2
        assert batch_state.step_count == 0
        assert batch_state.total_tokens_processed == 0
    
    def test_batch_update_creation(self):
        """Test BatchUpdate creation."""
        new_tokens = {1: [123, 456], 2: [789]}
        completed_sequences = [1]
        
        batch_update = BatchUpdate(
            new_tokens=new_tokens,
            completed_sequences=completed_sequences,
            inference_time=0.1,
            tokens_generated=3
        )
        
        assert batch_update.new_tokens == new_tokens
        assert batch_update.completed_sequences == completed_sequences
        assert batch_update.inference_time == 0.1
        assert batch_update.tokens_generated == 3


class TestContinuousBatchingScheduler:
    """Test continuous batching scheduler."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = ContinuousBatchingScheduler(
            max_queue_size=100,
            max_batch_size=4
        )
        
        assert scheduler.max_queue_size == 100
        assert scheduler.max_batch_size == 4
        assert len(scheduler.pending_requests) == 0
        assert len(scheduler.active_sequences) == 0
        assert scheduler.total_requests == 0
    
    def test_add_request(self):
        """Test adding requests to scheduler."""
        scheduler = ContinuousBatchingScheduler()
        sampling_params = SamplingParams(max_tokens=10)
        
        request = Request(
            request_id=1,
            prompt="Hello world",
            sampling_params=sampling_params
        )
        
        request_id = scheduler.add_request(request)
        
        assert len(scheduler.pending_requests) == 1
        assert scheduler.pending_requests[0].request_id == 1
        assert scheduler.total_requests == 1
    
    def test_get_batch_state_empty(self):
        """Test getting batch state when no active sequences."""
        scheduler = ContinuousBatchingScheduler()
        
        batch_state = scheduler.get_batch_state()
        
        assert batch_state is None
    
    def test_get_batch_state_with_sequences(self):
        """Test getting batch state with active sequences."""
        scheduler = ContinuousBatchingScheduler()
        sampling_params = SamplingParams(max_tokens=10)
        
        # Add a request and manually add it to active sequences
        request = Request(
            request_id=1,
            prompt="Hello world",
            sampling_params=sampling_params
        )
        
        seq_info = SequenceInfo(
            sequence_id=1,
            request_id=1,
            start_position=0,
            current_length=0,
            max_new_tokens=10,
            sampling_params=sampling_params
        )
        
        scheduler.active_sequences[1] = seq_info
        
        batch_state = scheduler.get_batch_state()
        
        assert batch_state is not None
        assert batch_state.batch_size == 1
        assert len(batch_state.active_sequences) == 1
    
    def test_update_batch(self):
        """Test updating batch with new tokens."""
        scheduler = ContinuousBatchingScheduler()
        sampling_params = SamplingParams(max_tokens=10)
        
        # Add a sequence
        seq_info = SequenceInfo(
            sequence_id=1,
            request_id=1,
            start_position=0,
            current_length=0,
            max_new_tokens=10,
            sampling_params=sampling_params
        )
        
        scheduler.active_sequences[1] = seq_info
        
        # Create batch update
        batch_update = BatchUpdate(
            new_tokens={1: [123, 456]},
            completed_sequences=[],
            inference_time=0.1,
            tokens_generated=2
        )
        
        scheduler.update_batch(batch_update)
        
        # Check that sequence was updated
        updated_seq = scheduler.active_sequences[1]
        assert updated_seq.current_length == 2
    
    def test_complete_sequence(self):
        """Test completing a sequence."""
        scheduler = ContinuousBatchingScheduler()
        sampling_params = SamplingParams(max_tokens=10)
        
        # Add a sequence
        seq_info = SequenceInfo(
            sequence_id=1,
            request_id=1,
            start_position=0,
            current_length=10,
            max_new_tokens=10,
            sampling_params=sampling_params
        )
        
        scheduler.active_sequences[1] = seq_info
        
        # Create batch update that completes the sequence
        batch_update = BatchUpdate(
            new_tokens={1: [123]},
            completed_sequences=[1],
            inference_time=0.1,
            tokens_generated=1
        )
        
        scheduler.update_batch(batch_update)
        
        # Check that sequence was completed
        assert 1 not in scheduler.active_sequences
        assert len(scheduler.completed_results) == 1
        assert scheduler.completed_requests_count == 1
    
    def test_get_stats(self):
        """Test getting scheduler statistics."""
        scheduler = ContinuousBatchingScheduler()
        
        stats = scheduler.get_stats()
        
        expected_keys = [
            "pending_requests",
            "active_sequences", 
            "completed_requests",
            "failed_requests",
            "total_requests",
            "batch_size",
            "has_batch"
        ]
        
        for key in expected_keys:
            assert key in stats


class TestRequest:
    """Test Request dataclass."""
    
    def test_request_creation(self):
        """Test Request creation."""
        sampling_params = SamplingParams(max_tokens=10)
        
        request = Request(
            request_id=1,
            prompt="Hello world",
            sampling_params=sampling_params
        )
        
        assert request.request_id == 1
        assert request.prompt == "Hello world"
        assert request.sampling_params == sampling_params
        assert request.priority == RequestPriority.NORMAL
        assert request.created_at is not None 