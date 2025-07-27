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
from .batch_state import BatchState, BatchUpdate, SequenceInfo
from .continuous_batching_scheduler import ContinuousBatchingScheduler, Request as CBRequest


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
        model_name: str = "/zx_data1/nano-vllm/models/Qwen3-0.6B",  # "Qwen/Qwen3-0.6B",
        device: str = "mps",
        dtype: str = "float16",
        max_queue_size: int = 1000,
        num_blocks: int = 1024,
        block_size: int = 16,
        max_seq_length: int = 4096,
        enable_batching: bool = True,
        max_batch_size: int = 8,
        enable_optimizations: bool = True,
        enable_fast_inference: bool = False,
        batching_mode: str = "static"  # "static" or "continuous"
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
            enable_batching: Whether to enable batch processing
            max_batch_size: Maximum batch size for processing
            enable_optimizations: Whether to enable model optimizations
            enable_fast_inference: Whether to enable aggressive optimizations
            batching_mode: Batching mode ("static" or "continuous")
        """
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.max_seq_length = max_seq_length
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self.batching_mode = batching_mode
        
        # Initialize components
        logger.info("Initializing LLM Engine components...")
        
        # Initialize block manager for memory management
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
            dtype=self.dtype
        )
        
        # Initialize scheduler based on batching mode
        if batching_mode == "continuous":
            self.scheduler = ContinuousBatchingScheduler(
                max_queue_size=max_queue_size,
                max_batch_size=max_batch_size
            )
            logger.info("Using continuous batching scheduler")
        else:
            self.scheduler = Scheduler(max_queue_size=max_queue_size)
            logger.info("Using static batching scheduler")
        
        # Initialize model runner for inference
        self.model_runner = ModelRunner(
            model_name=model_name,
            device=device,
            dtype=self.dtype,
            max_seq_length=max_seq_length
        )
        
        # Apply optimizations if enabled
        if enable_optimizations:
            self.model_runner.optimize_for_inference()
        
        if enable_fast_inference:
            self.model_runner.enable_fast_inference()
        
        # Performance tracking
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        logger.info(f"LLM Engine initialized with {model_name} on {self.model_runner.device}")
        if enable_batching:
            logger.info(f"Batch processing enabled with max_batch_size={max_batch_size}")
        if enable_optimizations:
            logger.info("Model optimizations enabled")
        if enable_fast_inference:
            logger.info("Fast inference mode enabled")
        logger.info(f"Batching mode: {batching_mode}")
    
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
        # Use continuous batching if enabled
        if self.batching_mode == "continuous":
            return self._generate_continuous(prompts, sampling_params, priority)
        else:
            return self._generate_static(prompts, sampling_params, priority)
    
    def _generate_static(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> List[Dict[str, Any]]:
        """
        Generate text using static batching (original implementation).
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
    
    def _generate_continuous(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> List[Dict[str, Any]]:
        """
        Generate text using continuous batching with structured batch states.
        
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
            request = CBRequest(
                request_id=len(request_ids),
                prompt=prompt,
                sampling_params=sampling_params,
                priority=priority
            )
            request_id = self.scheduler.add_request(request)
            request_ids.append(request_id)
        
        # Process until all requests complete
        results = []
        while request_ids:
            # Get current batch state
            batch_state = self.scheduler.get_batch_state()
            
            if batch_state is None:
                # No active sequences, try to add pending requests
                # This is handled by the scheduler internally
                time.sleep(0.001)  # Small delay to avoid busy waiting
                continue
            
            # Run model forward pass
            start_time = time.time()
            model_outputs = self.model_runner.run_model_batch(
                input_ids=batch_state.input_ids,
                attention_mask=batch_state.attention_mask
            )
            inference_time = time.time() - start_time
            
            # Sample next tokens for each sequence
            new_tokens = {}
            completed_sequences = []
            
            for sequence_id, seq_info in batch_state.sequence_map.items():
                if seq_info.is_complete:
                    continue
                
                # Get logits for this sequence
                sequence_logits = model_outputs["logits"][seq_info.start_position:seq_info.start_position+1]
                
                # Sample next token
                next_token = self._sample_next_token(sequence_logits, seq_info.sampling_params)
                
                if next_token is not None:
                    new_tokens[sequence_id] = [next_token.item()]
                    
                    # Check if sequence should complete
                    if (seq_info.current_length + 1 >= seq_info.max_new_tokens or
                        self._should_stop_generation(next_token, [], seq_info.sampling_params)):
                        completed_sequences.append(sequence_id)
                else:
                    # No token generated, sequence might be complete
                    completed_sequences.append(sequence_id)
            
            # Create batch update
            batch_update = BatchUpdate(
                new_tokens=new_tokens,
                completed_sequences=completed_sequences,
                model_outputs=model_outputs,
                inference_time=inference_time,
                tokens_generated=len(new_tokens)
            )
            
            # Update scheduler
            self.scheduler.update_batch(batch_update)
            
            # Check for completed results
            completed_results = self.scheduler.get_completed_results()
            for result in completed_results:
                if result['request_id'] in request_ids:
                    results.append(result)
                    request_ids.remove(result['request_id'])
        
        self.total_requests += len(prompts)
        self.completed_requests += len(results)
        return results
    
    def _sample_next_token(self, logits: torch.Tensor, sampling_params: SamplingParams) -> Optional[torch.Tensor]:
        """
        Sample next token from logits using sampling parameters.
        
        Args:
            logits: Logits for next token (1, vocab_size)
            sampling_params: Sampling parameters
            
        Returns:
            Sampled token ID or None if sampling failed
        """
        try:
            # Apply temperature
            if sampling_params.temperature > 0:
                logits = logits / sampling_params.temperature
            
            # Apply top-k sampling
            if sampling_params.top_k > 0:
                top_k = min(sampling_params.top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Apply top-p sampling
            if sampling_params.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > sampling_params.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            if sampling_params.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            return next_token
            
        except Exception as e:
            logger.error(f"Token sampling failed: {e}")
            return None
    
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
    
    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        max_batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Generate text for a batch of prompts efficiently.
        
        Args:
            prompts: List of prompts to process
            sampling_params: Sampling parameters for generation
            priority: Request priority
            max_batch_size: Maximum batch size for processing
            
        Returns:
            List of generation results
        """
        if not prompts:
            return []
        
        # Use default sampling parameters if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Add all requests to scheduler
        request_ids = []
        for prompt in prompts:
            request_id = self.scheduler.add_request(
                prompt=prompt,
                sampling_params=sampling_params,
                priority=priority
            )
            request_ids.append(request_id)
        
        # Process requests in batches
        results = []
        for i in range(0, len(request_ids), max_batch_size):
            batch_request_ids = request_ids[i:i + max_batch_size]
            batch_results = self._process_batch(batch_request_ids)
            results.extend(batch_results)
        
        self.total_requests += len(prompts)
        return results
    
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
    
    def _process_batch(self, request_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Process a batch of requests efficiently.
        
        Args:
            request_ids: List of request IDs to process
            
        Returns:
            List of generation results
        """
        # Get batch of requests from scheduler
        requests = []
        for request_id in request_ids:
            if request_id in self.scheduler.active_requests:
                requests.append(self.scheduler.active_requests[request_id])
        
        if not requests:
            return []
        
        # Prepare batch inputs
        batch_prompts = [req.prompt for req in requests]
        batch_sampling_params = [req.sampling_params for req in requests]
        
        # Tokenize all prompts
        batch_input_ids = []
        for prompt in batch_prompts:
            input_ids = self.model_runner.tokenize(prompt)
            batch_input_ids.append(input_ids)
        
        # Pad sequences to same length for batch processing
        max_length = max(ids.shape[1] for ids in batch_input_ids)
        padded_input_ids = []
        
        for input_ids in batch_input_ids:
            if input_ids.shape[1] < max_length:
                # Pad with tokenizer.pad_token_id or 0
                padding_length = max_length - input_ids.shape[1]
                padding = torch.full((1, padding_length), 0, dtype=input_ids.dtype, device=input_ids.device)
                padded_ids = torch.cat([input_ids, padding], dim=1)
            else:
                padded_ids = input_ids
            padded_input_ids.append(padded_ids)
        
        # Stack into batch tensor
        batch_tensor = torch.cat(padded_input_ids, dim=0)
        
        # Process batch through model
        try:
            batch_outputs = self.model_runner.run_model_batch(
                input_ids=batch_tensor,
                use_cache=True
            )
            
            # Process results for each request
            results = []
            for i, request in enumerate(requests):
                # Extract logits for this request
                request_logits = batch_outputs["logits"][i:i+1]
                
                # Generate text for this request
                generated_text, tokens_generated = self._generate_text_from_logits(
                    request_logits, 
                    batch_sampling_params[i]
                )
                
                result = {
                    "request_id": request.request_id,
                    "generated_text": generated_text,
                    "tokens_generated": tokens_generated,
                    "prompt": request.prompt
                }
                
                results.append(result)
                self.completed_requests += 1
                self.scheduler.mark_request_completed(request.request_id, result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fall back to individual processing
            return self._process_requests_individually(request_ids)
    
    def _process_requests_individually(self, request_ids: List[int]) -> List[Dict[str, Any]]:
        """Fallback to individual request processing."""
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
        return results
    
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
    
    def _generate_text_from_logits(
        self, 
        logits: torch.Tensor, 
        sampling_params: SamplingParams
    ) -> Tuple[str, int]:
        """
        Generate text from logits using sampling parameters.
        
        Args:
            logits: Model output logits
            sampling_params: Sampling parameters
            
        Returns:
            Tuple of (generated_text, tokens_generated)
        """
        generated_tokens = []
        current_logits = logits
        
        for _ in range(sampling_params.max_tokens):
            # Get logits for the last token
            next_token_logits = current_logits[:, -1, :]
            
            # Sample next token
            next_token_id = self.model_runner._sample_token(next_token_logits, sampling_params)
            generated_tokens.append(next_token_id.item())
            
            # Check if we should stop
            if self._should_stop_generation(next_token_id, generated_tokens, sampling_params):
                break
            
            # Prepare for next iteration (in a real implementation, you'd use KV cache)
            # For simplicity, we'll just continue with the current approach
            break  # Simplified for this example
        
        # Decode generated tokens
        if generated_tokens:
            generated_text = self.model_runner.detokenize(torch.tensor([generated_tokens]))
        else:
            generated_text = ""
        
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