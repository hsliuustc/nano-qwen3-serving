"""
LLM Engine - Central orchestrator for the nano LLM serving system.
"""

import time
import torch
from typing import Dict, List, Optional, Tuple, Any, Union, Generator
from loguru import logger

from .sampling_params import SamplingParams
from .model_runner import ModelRunner
from .scheduler import Scheduler, RequestPriority
from .block_manager import BlockManager


class LLMEngine:
    """
    Central orchestrator for the nano LLM serving engine.
    
    This class coordinates the interaction between all components:
    - Scheduler: Request management and prioritization
    - ModelRunner: Model execution and inference
    - BlockManager: Memory management and KV cache
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "mps",
        dtype: str = "float16",
        max_queue_size: int = 1000,
        num_blocks: int = 1024,
        block_size: int = 16,
        max_seq_length: int = 4096
    ):
        """
        Initialize the LLM engine.
        
        Args:
            model_name: Name of the model to load
            device: Device to run inference on
            dtype: Data type for model weights
            max_queue_size: Maximum number of requests in queue
            num_blocks: Number of memory blocks for KV cache
            block_size: Size of each memory block
            max_seq_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.max_seq_length = max_seq_length
        
        # Initialize components
        logger.info("Initializing LLM Engine components...")
        
        # Initialize block manager for memory management
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
            dtype=self.dtype
        )
        
        # Initialize scheduler for request management
        self.scheduler = Scheduler(max_queue_size=max_queue_size)
        
        # Initialize model runner for inference
        self.model_runner = ModelRunner(
            model_name=model_name,
            device=device,
            dtype=self.dtype,
            max_seq_length=max_seq_length
        )
        
        # Performance tracking
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        logger.info(f"LLM Engine initialized with {model_name} on {device}")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> List[Dict[str, Any]]:
        """
        Generate text for given prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            sampling_params: Sampling parameters for generation
            priority: Request priority
            
        Returns:
            List of generation results
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Use default sampling parameters if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Add requests to scheduler
        request_ids = []
        for prompt in prompts:
            request_id = self.scheduler.add_request(
                prompt=prompt,
                sampling_params=sampling_params,
                priority=priority
            )
            request_ids.append(request_id)
        
        # Process requests
        results = []
        for request_id in request_ids:
            try:
                result = self._process_request(request_id)
                results.append(result)
                self.completed_requests += 1
            except Exception as e:
                logger.error(f"Failed to process request {request_id}: {e}")
                self.scheduler.mark_request_failed(request_id, str(e))
                self.failed_requests += 1
                results.append({
                    "request_id": request_id,
                    "error": str(e),
                    "generated_text": "",
                    "tokens_generated": 0
                })
        
        self.total_requests += len(prompts)
        return results
    
    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate text with streaming output.
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters for generation
            priority: Request priority
            
        Yields:
            Streaming results with the following structure:
            {
                "token": str,              # Current token text
                "token_id": int,           # Token ID
                "text": str,               # Accumulated text so far
                "finished": bool,          # Whether generation is complete
                "tokens_generated": int,   # Number of tokens generated so far
                "request_id": int          # Request ID
            }
        """
        # Use default sampling parameters if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Add request to scheduler
        request_id = self.scheduler.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            priority=priority
        )
        
        self.total_requests += 1
        
        try:
            # Process streaming request
            for stream_result in self._process_streaming_request(request_id):
                yield stream_result
                if stream_result.get("finished", False):
                    self.completed_requests += 1
                    break
        except Exception as e:
            logger.error(f"Failed to process streaming request {request_id}: {e}")
            self.scheduler.mark_request_failed(request_id, str(e))
            self.failed_requests += 1
            yield {
                "request_id": request_id,
                "error": str(e),
                "token": "",
                "token_id": -1,
                "text": "",
                "finished": True,
                "tokens_generated": 0
            }
    
    def _process_streaming_request(self, request_id: int) -> Generator[Dict[str, Any], None, None]:
        """
        Process a single streaming request.
        
        Args:
            request_id: ID of the request to process
            
        Yields:
            Streaming generation results
        """
        # Get request from scheduler
        requests = self.scheduler.get_next_requests(batch_size=1)
        if not requests:
            raise RuntimeError(f"Request {request_id} not found in scheduler")
        
        request = requests[0]
        if request.request_id != request_id:
            raise RuntimeError(f"Request ID mismatch: expected {request_id}, got {request.request_id}")
        
        # Allocate memory blocks for the sequence
        sequence_id = request_id
        input_tokens = self.model_runner.tokenize(request.prompt)
        num_tokens = input_tokens.shape[1]
        
        # Allocate blocks for the sequence
        block_indices = self.block_manager.allocate_blocks(sequence_id, num_tokens)
        
        try:
            # Generate text with streaming
            accumulated_text = ""
            tokens_generated = 0
            
            for stream_result in self._generate_text_stream(
                input_tokens=input_tokens,
                sampling_params=request.sampling_params,
                sequence_id=sequence_id
            ):
                # Update accumulated text
                accumulated_text += stream_result["token"]
                tokens_generated += 1
                
                # Add additional metadata
                stream_result.update({
                    "text": accumulated_text,
                    "tokens_generated": tokens_generated,
                    "request_id": request_id,
                    "total_tokens": num_tokens + tokens_generated
                })
                
                yield stream_result
                
                # Check if generation is finished
                if stream_result.get("finished", False):
                    break
            
            # Mark request as completed
            final_result = {
                "request_id": request_id,
                "prompt": request.prompt,
                "generated_text": accumulated_text,
                "tokens_generated": tokens_generated,
                "total_tokens": num_tokens + tokens_generated,
                "block_indices": block_indices
            }
            self.scheduler.mark_request_completed(request_id, final_result)
            
        finally:
            # Free memory blocks
            self.block_manager.free_sequence_blocks(sequence_id)
    
    def _generate_text_stream(
        self,
        input_tokens: torch.Tensor,
        sampling_params: SamplingParams,
        sequence_id: int
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate text with streaming output.
        
        Args:
            input_tokens: Input token IDs
            sampling_params: Sampling parameters
            sequence_id: Sequence ID for tracking
            
        Yields:
            Streaming generation results
        """
        current_tokens = input_tokens.clone()
        past_key_values = None
        generated_tokens = []
        
        # Generation loop
        for step in range(sampling_params.max_tokens):
            # Use KV cache properly
            if past_key_values is not None:
                # When using cache, only pass the new token
                input_tokens = current_tokens[:, -1:]
            else:
                # First iteration: pass the full sequence
                input_tokens = current_tokens
            
            # Generate next token
            next_token, past_key_values = self.model_runner.generate_next_token(
                input_ids=input_tokens,
                sampling_params=sampling_params,
                past_key_values=past_key_values
            )
            
            # Add to generated tokens list
            generated_tokens.append(next_token.item())
            
            # Decode the token
            token_text = self.model_runner.detokenize(torch.tensor([next_token.item()]))
            
            # Check for stop conditions
            should_stop = self._should_stop_generation(next_token, generated_tokens, sampling_params)
            
            # Yield streaming result
            yield {
                "token": token_text,
                "token_id": next_token.item(),
                "finished": should_stop
            }
            
            # Stop if needed
            if should_stop:
                break
            
            # Update current tokens for next iteration
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Yield final result if we reached max tokens
        if step == sampling_params.max_tokens - 1:
            yield {
                "token": "",
                "token_id": -1,
                "finished": True
            }
    
    def _process_request(self, request_id: int) -> Dict[str, Any]:
        """
        Process a single request through the generation pipeline.
        
        Args:
            request_id: ID of the request to process
            
        Returns:
            Generation result
        """
        # Get request from scheduler
        requests = self.scheduler.get_next_requests(batch_size=1)
        if not requests:
            raise RuntimeError(f"Request {request_id} not found in scheduler")
        
        request = requests[0]
        if request.request_id != request_id:
            raise RuntimeError(f"Request ID mismatch: expected {request_id}, got {request.request_id}")
        
        # Allocate memory blocks for the sequence
        sequence_id = request_id
        input_tokens = self.model_runner.tokenize(request.prompt)
        num_tokens = input_tokens.shape[1]
        
        # Allocate blocks for the sequence
        block_indices = self.block_manager.allocate_blocks(sequence_id, num_tokens)
        
        try:
            # Generate text
            generated_text, tokens_generated = self._generate_text(
                input_tokens=input_tokens,
                sampling_params=request.sampling_params,
                sequence_id=sequence_id
            )
            
            result = {
                "request_id": request_id,
                "prompt": request.prompt,
                "generated_text": generated_text,
                "tokens_generated": tokens_generated,
                "total_tokens": num_tokens + tokens_generated,
                "block_indices": block_indices
            }
            
            # Mark request as completed
            self.scheduler.mark_request_completed(request_id, result)
            
            return result
            
        finally:
            # Free memory blocks
            self.block_manager.free_sequence_blocks(sequence_id)
    
    def _generate_text(
        self,
        input_tokens: torch.Tensor,
        sampling_params: SamplingParams,
        sequence_id: int
    ) -> Tuple[str, int]:
        """
        Generate text using the model.
        
        Args:
            input_tokens: Input token IDs
            sampling_params: Sampling parameters
            sequence_id: Sequence ID for tracking
            
        Returns:
            Tuple of (generated_text, tokens_generated)
        """
        current_tokens = input_tokens.clone()
        generated_tokens = []
        past_key_values = None
        
        # Generation loop
        for step in range(sampling_params.max_tokens):
            # Use KV cache properly
            if past_key_values is not None:
                # When using cache, only pass the new token
                input_tokens = current_tokens[:, -1:]
            else:
                # First iteration: pass the full sequence
                input_tokens = current_tokens
            
            # Generate next token
            next_token, past_key_values = self.model_runner.generate_next_token(
                input_ids=input_tokens,
                sampling_params=sampling_params,
                past_key_values=past_key_values
            )
            
            # Add to generated tokens
            generated_tokens.append(next_token.item())
            
            # Check for stop conditions
            if self._should_stop_generation(next_token, generated_tokens, sampling_params):
                break
            
            # Update current tokens for next iteration
            # Concatenate the new token to the existing sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Decode generated tokens
        generated_text = self.model_runner.detokenize(torch.tensor(generated_tokens))
        
        return generated_text, len(generated_tokens)
    
    def _should_stop_generation(
        self,
        next_token: torch.Tensor,
        generated_tokens: List[int],
        sampling_params: SamplingParams
    ) -> bool:
        """
        Check if generation should stop.
        
        Args:
            next_token: Next generated token
            generated_tokens: List of generated tokens so far
            sampling_params: Sampling parameters
            
        Returns:
            True if generation should stop
        """
        # Check stop token IDs
        if sampling_params.stop_token_ids and next_token.item() in sampling_params.stop_token_ids:
            return True
        
        # Check stop sequences (simplified implementation)
        if sampling_params.stop_sequences:
            current_text = self.model_runner.detokenize(torch.tensor(generated_tokens))
            for stop_seq in sampling_params.stop_sequences:
                if stop_seq in current_text:
                    return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        uptime = time.time() - self.start_time
        
        return {
            "uptime": uptime,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.completed_requests / max(self.total_requests, 1),
            "requests_per_second": self.total_requests / max(uptime, 1),
            "scheduler_stats": self.scheduler.get_queue_stats(),
            "model_stats": self.model_runner.get_performance_stats(),
            "memory_stats": self.block_manager.get_memory_stats()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model_runner.get_model_info()
    
    def clear_stats(self) -> None:
        """Clear all statistics."""
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        self.scheduler.clear_completed_requests()
        self.model_runner.clear_performance_stats()
        
        logger.info("All statistics cleared")
    
    def shutdown(self) -> None:
        """Shutdown the engine and free resources."""
        logger.info("Shutting down LLM Engine...")
        
        # Clear all completed requests
        self.scheduler.clear_completed_requests(max_age=0)
        
        # Clear performance stats
        self.model_runner.clear_performance_stats()
        
        logger.info("LLM Engine shutdown complete") 