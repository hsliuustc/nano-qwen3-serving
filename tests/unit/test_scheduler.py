"""
Tests for Scheduler class.
"""

import pytest
import time
from nano_qwen3_serving.core.scheduler import Scheduler, Request, RequestPriority


class TestRequest:
    """Test cases for Request."""
    
    def test_request_creation(self):
        """Test Request creation."""
        request = Request(
            request_id=1,
            prompt="Hello",
            sampling_params=None,
            priority=RequestPriority.NORMAL
        )
        
        assert request.request_id == 1
        assert request.prompt == "Hello"
        assert request.sampling_params is None
        assert request.priority == RequestPriority.NORMAL
        assert request.max_wait_time == 30.0
        assert request.sequence_id is None
    
    def test_request_comparison(self):
        """Test Request comparison for priority queue."""
        # Higher priority should come first
        high_priority = Request(1, "High", None, RequestPriority.HIGH)
        normal_priority = Request(2, "Normal", None, RequestPriority.NORMAL)
        
        assert high_priority < normal_priority
        
        # Same priority, earlier creation time should come first
        early_request = Request(1, "Early", None, RequestPriority.NORMAL)
        time.sleep(0.01)  # Ensure different creation times
        late_request = Request(2, "Late", None, RequestPriority.NORMAL)
        
        assert early_request < late_request


class TestScheduler:
    """Test cases for Scheduler."""
    
    def test_initialization(self):
        """Test Scheduler initialization."""
        scheduler = Scheduler(max_queue_size=100)
        
        assert scheduler.max_queue_size == 100
        assert len(scheduler.request_queue) == 0
        assert len(scheduler.active_requests) == 0
        assert len(scheduler.completed_requests) == 0
        assert scheduler.next_request_id == 0
        assert scheduler.total_requests == 0
        assert scheduler.completed_requests_count == 0
        assert scheduler.failed_requests_count == 0
    
    def test_add_request(self):
        """Test adding requests to the queue."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Add a request
        request_id = scheduler.add_request(
            prompt="Hello",
            sampling_params=None,
            priority=RequestPriority.NORMAL
        )
        
        assert request_id == 0
        assert len(scheduler.request_queue) == 1
        assert scheduler.total_requests == 1
        
        # Add another request
        request_id2 = scheduler.add_request(
            prompt="World",
            sampling_params=None,
            priority=RequestPriority.HIGH
        )
        
        assert request_id2 == 1
        assert len(scheduler.request_queue) == 2
        assert scheduler.total_requests == 2
    
    def test_queue_full(self):
        """Test behavior when queue is full."""
        scheduler = Scheduler(max_queue_size=1)
        
        # Add one request (should succeed)
        scheduler.add_request("First", None)
        
        # Try to add another request (should fail)
        with pytest.raises(RuntimeError):
            scheduler.add_request("Second", None)
    
    def test_get_next_requests(self):
        """Test getting next requests from the queue."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Add requests with different priorities
        scheduler.add_request("Low", None, RequestPriority.LOW)
        scheduler.add_request("High", None, RequestPriority.HIGH)
        scheduler.add_request("Normal", None, RequestPriority.NORMAL)
        
        # Get next request (should be highest priority)
        requests = scheduler.get_next_requests(batch_size=1)
        assert len(requests) == 1
        assert requests[0].prompt == "High"
        assert requests[0].priority == RequestPriority.HIGH
        
        # Get next request (should be normal priority)
        requests = scheduler.get_next_requests(batch_size=1)
        assert len(requests) == 1
        assert requests[0].prompt == "Normal"
        
        # Get next request (should be low priority)
        requests = scheduler.get_next_requests(batch_size=1)
        assert len(requests) == 1
        assert requests[0].prompt == "Low"
    
    def test_get_next_requests_batch(self):
        """Test getting multiple requests at once."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Add multiple requests
        for i in range(3):
            scheduler.add_request(f"Request {i}", None)
        
        # Get all requests at once
        requests = scheduler.get_next_requests(batch_size=3)
        assert len(requests) == 3
        
        # Check that they're in the active requests
        for request in requests:
            assert request.request_id in scheduler.active_requests
    
    def test_mark_request_completed(self):
        """Test marking requests as completed."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Add and get a request
        request_id = scheduler.add_request("Test", None)
        requests = scheduler.get_next_requests(batch_size=1)
        assert len(requests) == 1
        
        # Mark as completed
        result = {"text": "Generated response"}
        scheduler.mark_request_completed(request_id, result)
        
        assert request_id not in scheduler.active_requests
        assert request_id in scheduler.completed_requests
        assert scheduler.completed_requests_count == 1
        assert scheduler.failed_requests_count == 0
        
        # Check the result
        completed = scheduler.completed_requests[request_id]
        assert completed["result"] == result
        assert "completed_at" in completed
    
    def test_mark_request_failed(self):
        """Test marking requests as failed."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Add and get a request
        request_id = scheduler.add_request("Test", None)
        requests = scheduler.get_next_requests(batch_size=1)
        
        # Mark as failed
        error = "Model loading failed"
        scheduler.mark_request_failed(request_id, error)
        
        assert request_id not in scheduler.active_requests
        assert request_id in scheduler.completed_requests
        assert scheduler.completed_requests_count == 0
        assert scheduler.failed_requests_count == 1
        
        # Check the error
        failed = scheduler.completed_requests[request_id]
        assert failed["error"] == error
        assert "failed_at" in failed
    
    def test_get_request_status(self):
        """Test getting request status."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Add a request
        request_id = scheduler.add_request("Test", None)
        
        # Check status before processing
        status = scheduler.get_request_status(request_id)
        assert status is None  # Not in active or completed yet
        
        # Get the request (makes it active)
        scheduler.get_next_requests(batch_size=1)
        status = scheduler.get_request_status(request_id)
        assert status["status"] == "active"
        assert status["request_id"] == request_id
        
        # Mark as completed
        scheduler.mark_request_completed(request_id, {"text": "Done"})
        status = scheduler.get_request_status(request_id)
        assert status["status"] == "completed"
        assert status["request_id"] == request_id
    
    def test_get_queue_stats(self):
        """Test getting queue statistics."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Get initial stats
        stats = scheduler.get_queue_stats()
        assert stats["queue_size"] == 0
        assert stats["active_requests"] == 0
        assert stats["completed_requests"] == 0
        assert stats["failed_requests"] == 0
        assert stats["total_requests"] == 0
        assert stats["average_wait_time"] == 0.0
        
        # Add and process some requests
        request_id1 = scheduler.add_request("Request 1", None)
        request_id2 = scheduler.add_request("Request 2", None)
        
        # Get both requests
        scheduler.get_next_requests(batch_size=2)
        
        # Mark one as completed, one as failed
        scheduler.mark_request_completed(request_id1, {"text": "Done"})
        scheduler.mark_request_failed(request_id2, "Error")
        
        # Get updated stats
        stats = scheduler.get_queue_stats()
        assert stats["queue_size"] == 0
        assert stats["active_requests"] == 0
        assert stats["completed_requests"] == 1
        assert stats["failed_requests"] == 1
        assert stats["total_requests"] == 2
    
    def test_expired_requests(self):
        """Test handling of expired requests."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Add a request with very short timeout
        request_id = scheduler.add_request("Test", None, max_wait_time=0.001)
        
        # Wait for it to expire
        time.sleep(0.01)
        
        # Try to get requests (expired ones should be removed)
        requests = scheduler.get_next_requests(batch_size=1)
        assert len(requests) == 0
        
        # Check that it was marked as failed
        assert scheduler.failed_requests_count == 1
    
    def test_clear_completed_requests(self):
        """Test clearing old completed requests."""
        scheduler = Scheduler(max_queue_size=5)
        
        # Add and complete a request
        request_id = scheduler.add_request("Test", None)
        scheduler.get_next_requests(batch_size=1)
        scheduler.mark_request_completed(request_id, {"text": "Done"})
        
        # Initially should have one completed request
        assert len(scheduler.completed_requests) == 1
        
        # Clear with very short max age
        scheduler.clear_completed_requests(max_age=0.001)
        
        # Wait and clear again
        time.sleep(0.01)
        scheduler.clear_completed_requests(max_age=0.001)
        
        # Should be cleared
        assert len(scheduler.completed_requests) == 0
    
    def test_request_priority_ordering(self):
        """Test that requests are properly ordered by priority."""
        scheduler = Scheduler(max_queue_size=10)
        
        # Add requests in random order
        scheduler.add_request("Low", None, RequestPriority.LOW)
        scheduler.add_request("Urgent", None, RequestPriority.URGENT)
        scheduler.add_request("Normal", None, RequestPriority.NORMAL)
        scheduler.add_request("High", None, RequestPriority.HIGH)
        
        # Get all requests
        requests = scheduler.get_next_requests(batch_size=4)
        
        # Check priority order (highest first)
        expected_order = ["Urgent", "High", "Normal", "Low"]
        actual_order = [req.prompt for req in requests]
        assert actual_order == expected_order
    
    def test_concurrent_request_handling(self):
        """Test handling multiple concurrent requests."""
        scheduler = Scheduler(max_queue_size=10)
        
        # Add multiple requests
        request_ids = []
        for i in range(5):
            request_id = scheduler.add_request(f"Request {i}", None)
            request_ids.append(request_id)
        
        # Get all requests
        requests = scheduler.get_next_requests(batch_size=5)
        assert len(requests) == 5
        
        # Mark some as completed, some as failed
        for i, request_id in enumerate(request_ids):
            if i % 2 == 0:
                scheduler.mark_request_completed(request_id, {"text": f"Result {i}"})
            else:
                scheduler.mark_request_failed(request_id, f"Error {i}")
        
        # Check final stats
        stats = scheduler.get_queue_stats()
        assert stats["completed_requests"] == 3  # 0, 2, 4
        assert stats["failed_requests"] == 2    # 1, 3
        assert stats["total_requests"] == 5 