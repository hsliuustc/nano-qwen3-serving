"""
Async LLM Engine - Async wrapper around LLMEngine for concurrent request handling.

This module provides async/await support for the nano LLM serving engine by wrapping
the existing synchronous components with async request handling and coordination.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union, AsyncGenerator
from loguru import logger

from ..core.engine import LLMEngine
from ..core.sampling_params import SamplingParams
from ..core.scheduler import RequestPriority


class AsyncLLMEngine:
    """
    Async wrapper around LLMEngine for concurrent request handling.
    
    This class provides async/await support by:
    1. Using the existing LLMEngine for all model operations
    2. Adding async request queuing and coordination
    3. Managing concurrent requests with worker tasks
    4. Providing async streaming support
    
    Key benefits:
    - Reuses proven synchronous components
    - Minimal code changes and complexity
    - Better performance (no async overhead on GPU operations)
    - Gradual migration path
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "mps",
        dtype: str = "float16",
        max_queue_size: int = 1000,
        num_blocks: int = 1024,
        block_size: int = 16,
        max_seq_length: int = 4096,
        worker_count: int = 2
    ):
        """
        Initialize the async LLM engine.
        
        Args:
            model_name: Name of the model to load
            device: Device to run inference on
            dtype: Data type for model weights
            max_queue_size: Maximum number of requests in queue
            num_blocks: Number of memory blocks for KV cache
            block_size: Size of each memory block
            max_seq_length: Maximum sequence length
            worker_count: Number of async worker tasks
        """
        # Initialize the underlying sync engine
        self.sync_engine = LLMEngine(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_queue_size=max_queue_size,
            num_blocks=num_blocks,
            block_size=block_size,
            max_seq_length=max_seq_length
        )
        
        # Async-specific components
        self.worker_count = worker_count
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        # Performance tracking
        self.async_stats = {
            "total_async_requests": 0,
            "completed_async_requests": 0,
            "failed_async_requests": 0,
            "average_wait_time": 0.0,
            "start_time": time.time()
        }
        
        logger.info(f"AsyncLLMEngine initialized with {worker_count} workers")
    
    async def start(self) -> None:
        """
        Start the async workers and begin processing requests.
        """
        if self.running:
            logger.warning("AsyncLLMEngine is already running")
            return
        
        logger.info(f"Starting {self.worker_count} async workers...")
        self.running = True
        
        # Start worker tasks
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info("AsyncLLMEngine started successfully")
    
    async def stop(self) -> None:
        """
        Stop the async workers and shutdown gracefully.
        """
        if not self.running:
            logger.warning("AsyncLLMEngine is not running")
            return
        
        logger.info("Stopping async workers...")
        self.running = False
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("AsyncLLMEngine stopped")
    
    async def generate_async(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> List[Dict[str, Any]]:
        """
        Generate text asynchronously for given prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            sampling_params: Sampling parameters for generation
            priority: Request priority
            
        Returns:
            List of generation results
        """
        if not self.running:
            await self.start()
        
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Submit all requests
        futures = []
        for prompt in prompts:
            future = asyncio.Future()
            await self.request_queue.put(('generate', prompt, sampling_params, priority, future))
            futures.append(future)
        
        # Wait for all results
        results = await asyncio.gather(*futures)
        return results
    
    async def generate_stream_async(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate text with async streaming output.
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters for generation
            priority: Request priority
            
        Yields:
            Streaming results
        """
        if not self.running:
            await self.start()
        
        # Create a queue for streaming results
        stream_queue = asyncio.Queue()
        future = asyncio.Future()
        
        # Submit streaming request
        await self.request_queue.put(('stream', prompt, sampling_params, priority, future, stream_queue))
        
        # Wait for streaming to start
        await future
        
        # Yield streaming results
        while True:
            try:
                result = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                if result.get('finished', False):
                    break
                yield result
            except asyncio.TimeoutError:
                # Check if the future is done (streaming finished)
                if future.done():
                    break
                continue
    
    async def submit_request_async(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> int:
        """
        Submit a request and return request ID for later retrieval.
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            priority: Request priority
            
        Returns:
            Request ID
        """
        if not self.running:
            await self.start()
        
        # Create a future for this request
        future = asyncio.Future()
        request_id = id(future)
        
        # Submit to queue
        await self.request_queue.put(('generate', prompt, sampling_params, priority, future))
        
        return request_id
    
    async def get_result_async(self, request_id: int) -> Optional[Dict[str, Any]]:
        """
        Get result for a submitted request.
        
        Args:
            request_id: Request ID from submit_request_async
            
        Returns:
            Generation result or None if not ready
        """
        # This is a simplified implementation
        # In a real implementation, you'd need to track futures by ID
        raise NotImplementedError("get_result_async not implemented yet")
    
    async def generate_batch_async(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        batch_size: int = 4,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> List[Dict[str, Any]]:
        """
        Generate text for a batch of prompts efficiently.
        
        Args:
            prompts: List of prompts
            sampling_params: Sampling parameters
            batch_size: Number of prompts to process simultaneously
            priority: Request priority
            
        Returns:
            List of generation results
        """
        # Process in batches
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = await self.generate_async(batch, sampling_params, priority)
            results.extend(batch_results)
        
        return results
    
    async def _worker_loop(self, worker_name: str) -> None:
        """
        Worker loop that processes requests using the sync engine.
        
        Args:
            worker_name: Name of the worker for logging
        """
        logger.debug(f"{worker_name} started")
        
        while self.running:
            try:
                # Get request from queue
                item = await self.request_queue.get()
                
                if item[0] == 'generate':
                    await self._handle_generate_request(worker_name, item[1:])
                elif item[0] == 'stream':
                    await self._handle_stream_request(worker_name, item[1:])
                else:
                    logger.error(f"{worker_name}: Unknown request type {item[0]}")
                
                # Mark task as done
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                logger.debug(f"{worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
                # Continue processing other requests
        
        logger.debug(f"{worker_name} stopped")
    
    async def _handle_generate_request(
        self,
        worker_name: str,
        item: Tuple[str, Optional[SamplingParams], RequestPriority, asyncio.Future]
    ) -> None:
        """
        Handle a generate request.
        
        Args:
            worker_name: Name of the worker
            item: Request item (prompt, sampling_params, priority, future)
        """
        prompt, sampling_params, priority, future = item
        
        try:
            start_time = time.time()
            
            # Use sync engine to generate
            result = self.sync_engine.generate(prompt, sampling_params, priority)
            
            # Update stats
            wait_time = time.time() - start_time
            self.async_stats["completed_async_requests"] += 1
            self.async_stats["average_wait_time"] = (
                (self.async_stats["average_wait_time"] * (self.async_stats["completed_async_requests"] - 1) + wait_time) /
                self.async_stats["completed_async_requests"]
            )
            
            # Set result
            future.set_result(result[0] if isinstance(result, list) else result)
            
            logger.debug(f"{worker_name} completed request in {wait_time:.3f}s")
            
        except Exception as e:
            logger.error(f"{worker_name} failed to process request: {e}")
            self.async_stats["failed_async_requests"] += 1
            future.set_exception(e)
    
    async def _handle_stream_request(
        self,
        worker_name: str,
        item: Tuple[str, Optional[SamplingParams], RequestPriority, asyncio.Future, asyncio.Queue]
    ) -> None:
        """
        Handle a streaming request.
        
        Args:
            worker_name: Name of the worker
            item: Request item (prompt, sampling_params, priority, future, stream_queue)
        """
        prompt, sampling_params, priority, future, stream_queue = item
        
        try:
            start_time = time.time()
            
            # Signal that streaming is starting
            future.set_result(None)
            
            # Use sync engine to generate stream
            for result in self.sync_engine.generate_stream(prompt, sampling_params, priority):
                await stream_queue.put(result)
                if result.get('finished', False):
                    break
            
            # Update stats
            wait_time = time.time() - start_time
            self.async_stats["completed_async_requests"] += 1
            self.async_stats["average_wait_time"] = (
                (self.async_stats["average_wait_time"] * (self.async_stats["completed_async_requests"] - 1) + wait_time) /
                self.async_stats["completed_async_requests"]
            )
            
            logger.debug(f"{worker_name} completed streaming request in {wait_time:.3f}s")
            
        except Exception as e:
            logger.error(f"{worker_name} failed to process streaming request: {e}")
            self.async_stats["failed_async_requests"] += 1
            if not future.done():
                future.set_exception(e)
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """
        Get async engine statistics.
        
        Returns:
            Dictionary containing async engine stats
        """
        sync_stats = self.sync_engine.get_stats()
        
        return {
            "async_stats": self.async_stats,
            "sync_stats": sync_stats,
            "queue_size": self.request_queue.qsize(),
            "worker_count": len(self.workers),
            "running": self.running
        }
    
    async def clear_stats_async(self) -> None:
        """
        Clear async engine statistics.
        """
        self.async_stats = {
            "total_async_requests": 0,
            "completed_async_requests": 0,
            "failed_async_requests": 0,
            "average_wait_time": 0.0,
            "start_time": time.time()
        }
        self.sync_engine.clear_stats()
    
    async def shutdown_async(self) -> None:
        """
        Shutdown the async engine gracefully.
        """
        await self.stop()
        self.sync_engine.shutdown()
        logger.info("AsyncLLMEngine shutdown complete")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        """
        return self.sync_engine.get_model_info()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown_async() 