"""
Async LLM - High-level async interface for the nano LLM serving engine.

This module provides a simple async/await API for text generation using the
AsyncLLMEngine under the hood.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from loguru import logger

from .async_engine import AsyncLLMEngine
from ..core.sampling_params import SamplingParams
from ..core.scheduler import RequestPriority


class AsyncLLM:
    """
    High-level async interface for the nano LLM serving engine.
    
    This class provides a simple async/await API for text generation,
    similar to the synchronous LLM class but with async support for
    concurrent request handling.
    
    Example usage:
        async with AsyncLLM() as llm:
            result = await llm.generate("Hello, world!")
            print(result["generated_text"])
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
        Initialize the async LLM.
        
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
        self.async_engine = AsyncLLMEngine(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_queue_size=max_queue_size,
            num_blocks=num_blocks,
            block_size=block_size,
            max_seq_length=max_seq_length,
            worker_count=worker_count
        )
        
        logger.info(f"AsyncLLM initialized with {model_name} on {device}")
    
    async def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate text asynchronously.
        
        Args:
            prompts: Single prompt or list of prompts
            sampling_params: Sampling parameters for generation
            priority: Request priority
            
        Returns:
            Generation result(s)
        """
        results = await self.async_engine.generate_async(prompts, sampling_params, priority)
        
        # Return single result for single prompt
        if isinstance(prompts, str):
            return results[0]
        return results
    
    async def generate_stream(
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
        async for result in self.async_engine.generate_stream_async(prompt, sampling_params, priority):
            yield result
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> Dict[str, Any]:
        """
        Generate chat response asynchronously.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            sampling_params: Sampling parameters for generation
            priority: Request priority
            
        Returns:
            Chat response
        """
        # Format messages as prompt
        prompt = self._format_chat_prompt(messages)
        
        result = await self.generate(prompt, sampling_params, priority)
        return result
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate chat response with async streaming output.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            sampling_params: Sampling parameters for generation
            priority: Request priority
            
        Yields:
            Streaming chat results
        """
        # Format messages as prompt
        prompt = self._format_chat_prompt(messages)
        
        async for result in self.generate_stream(prompt, sampling_params, priority):
            yield result
    
    async def generate_batch(
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
        return await self.async_engine.generate_batch_async(prompts, sampling_params, batch_size, priority)
    
    async def submit_request(
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
        return await self.async_engine.submit_request_async(prompt, sampling_params, priority)
    
    async def get_result(self, request_id: int) -> Optional[Dict[str, Any]]:
        """
        Get result for a submitted request.
        
        Args:
            request_id: Request ID from submit_request
            
        Returns:
            Generation result or None if not ready
        """
        return await self.async_engine.get_result_async(request_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get LLM statistics.
        
        Returns:
            Dictionary containing LLM stats
        """
        return await self.async_engine.get_stats_async()
    
    async def clear_stats(self) -> None:
        """
        Clear LLM statistics.
        """
        await self.async_engine.clear_stats_async()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        """
        return self.async_engine.get_model_info()
    
    async def start(self) -> None:
        """
        Start the async LLM engine.
        """
        await self.async_engine.start()
    
    async def shutdown(self) -> None:
        """
        Shutdown the async LLM engine.
        """
        await self.async_engine.shutdown_async()
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages as a prompt.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
            elif role == 'system':
                prompt_parts.append(f"System: {content}")
        
        # Add final assistant prefix
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown() 