"""
Request scheduling and management for the LLM engine.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import heapq


class RequestPriority(Enum):
    """Priority levels for requests."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Request:
    """Represents a generation request."""
    request_id: int
    prompt: str
    sampling_params: Any  # SamplingParams
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    max_wait_time: float = 30.0  # seconds
    sequence_id: Optional[int] = None
    
    def __lt__(self, other):
        """Compare requests for priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at


class Scheduler:
    """
    Manages request scheduling and prioritization.
    
    This class handles the queuing, prioritization, and scheduling of generation
    requests to optimize throughput and fairness.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize the scheduler.
        
        Args:
            max_queue_size: Maximum number of requests in the queue
        """
        self.max_queue_size = max_queue_size
        self.request_queue: List[Request] = []
        self.active_requests: Dict[int, Request] = {}
        self.completed_requests: Dict[int, Any] = {}
        self.next_request_id = 0
        
        # Statistics
        self.total_requests = 0
        self.completed_requests_count = 0
        self.failed_requests_count = 0
        
        logger.info(f"Scheduler initialized with max queue size {max_queue_size}")
    
    def add_request(
        self,
        prompt: str,
        sampling_params: Any,
        priority: RequestPriority = RequestPriority.NORMAL,
        max_wait_time: float = 30.0
    ) -> int:
        """
        Add a new request to the queue.
        
        Args:
            prompt: Input prompt for generation
            sampling_params: Sampling parameters
            priority: Request priority
            max_wait_time: Maximum time to wait in queue
            
        Returns:
            Request ID
        """
        if len(self.request_queue) >= self.max_queue_size:
            raise RuntimeError("Request queue is full")
        
        request_id = self._get_next_request_id()
        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            priority=priority,
            max_wait_time=max_wait_time
        )
        
        heapq.heappush(self.request_queue, request)
        self.total_requests += 1
        
        logger.debug(f"Added request {request_id} to queue (priority: {priority.name})")
        return request_id
    
    def get_next_requests(self, batch_size: int = 1) -> List[Request]:
        """
        Get the next batch of requests to process.
        
        Args:
            batch_size: Number of requests to return
            
        Returns:
            List of requests to process
        """
        requests = []
        current_time = time.time()
        
        # Remove expired requests
        self._remove_expired_requests(current_time)
        
        # Get requests from queue
        while len(requests) < batch_size and self.request_queue:
            request = heapq.heappop(self.request_queue)
            
            # Check if request has expired
            if current_time - request.created_at > request.max_wait_time:
                logger.warning(f"Request {request.request_id} expired")
                self.failed_requests_count += 1
                continue
            
            requests.append(request)
            self.active_requests[request.request_id] = request
        
        logger.debug(f"Scheduled {len(requests)} requests for processing")
        return requests
    
    def mark_request_completed(self, request_id: int, result: Any) -> None:
        """
        Mark a request as completed.
        
        Args:
            request_id: ID of the completed request
            result: Generation result
        """
        if request_id in self.active_requests:
            request = self.active_requests.pop(request_id)
            self.completed_requests[request_id] = {
                "request": request,
                "result": result,
                "completed_at": time.time()
            }
            self.completed_requests_count += 1
            
            logger.debug(f"Request {request_id} completed")
    
    def mark_request_failed(self, request_id: int, error: str) -> None:
        """
        Mark a request as failed.
        
        Args:
            request_id: ID of the failed request
            error: Error message
        """
        if request_id in self.active_requests:
            request = self.active_requests.pop(request_id)
            self.completed_requests[request_id] = {
                "request": request,
                "error": error,
                "failed_at": time.time()
            }
            self.failed_requests_count += 1
            
            logger.error(f"Request {request_id} failed: {error}")
    
    def get_request_status(self, request_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the status of a request.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Request status information
        """
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                "status": "active",
                "request_id": request_id,
                "created_at": request.created_at,
                "priority": request.priority.name
            }
        elif request_id in self.completed_requests:
            result = self.completed_requests[request_id]
            if "error" in result:
                return {
                    "status": "failed",
                    "request_id": request_id,
                    "error": result["error"],
                    "failed_at": result["failed_at"]
                }
            else:
                return {
                    "status": "completed",
                    "request_id": request_id,
                    "completed_at": result["completed_at"]
                }
        else:
            return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        current_time = time.time()
        self._remove_expired_requests(current_time)
        
        return {
            "queue_size": len(self.request_queue),
            "active_requests": len(self.active_requests),
            "completed_requests": self.completed_requests_count,
            "failed_requests": self.failed_requests_count,
            "total_requests": self.total_requests,
            "average_wait_time": self._calculate_average_wait_time()
        }
    
    def _get_next_request_id(self) -> int:
        """Get the next available request ID."""
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id
    
    def _remove_expired_requests(self, current_time: float) -> None:
        """Remove expired requests from the queue."""
        expired_count = 0
        temp_queue = []
        
        while self.request_queue:
            request = heapq.heappop(self.request_queue)
            if current_time - request.created_at <= request.max_wait_time:
                heapq.heappush(temp_queue, request)
            else:
                expired_count += 1
                self.failed_requests_count += 1
        
        self.request_queue = temp_queue
        
        if expired_count > 0:
            logger.warning(f"Removed {expired_count} expired requests from queue")
    
    def _calculate_average_wait_time(self) -> float:
        """Calculate average wait time for completed requests."""
        if not self.completed_requests:
            return 0.0
        
        total_wait_time = 0.0
        count = 0
        
        for result in self.completed_requests.values():
            if "completed_at" in result:
                wait_time = result["completed_at"] - result["request"].created_at
                total_wait_time += wait_time
                count += 1
        
        return total_wait_time / count if count > 0 else 0.0
    
    def clear_completed_requests(self, max_age: float = 3600.0) -> None:
        """
        Clear old completed requests to free memory.
        
        Args:
            max_age: Maximum age of requests to keep (in seconds)
        """
        current_time = time.time()
        to_remove = []
        
        for request_id, result in self.completed_requests.items():
            if "completed_at" in result:
                age = current_time - result["completed_at"]
            elif "failed_at" in result:
                age = current_time - result["failed_at"]
            else:
                continue
            
            if age > max_age:
                to_remove.append(request_id)
        
        for request_id in to_remove:
            del self.completed_requests[request_id]
        
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old completed requests") 