"""
Main LLM class - High-level interface for text generation.
"""

from typing import Dict, List, Optional, Union, Any, Iterator, Generator
from loguru import logger

from .sampling_params import SamplingParams
from .engine import LLMEngine
from .scheduler import RequestPriority


class LLM:
    """
    Main inference class for the nano LLM serving engine.
    
    This class provides the high-level interface for text generation,
    following the workflow from the sequence diagram:
    1. User calls generate(prompts, sampling_params)
    2. LLM calls LLMEngine.generate()
    3. LLMEngine orchestrates the generation process
    4. Returns output dictionaries to user
    """
    
    def __init__(
        self,
        model_name: str = "/zx_data1/nano-vllm/models/Qwen3-0.6B",  # "Qwen/Qwen3-0.6B",
        device: str = "auto",
        dtype: Optional[str] = None,
        max_queue_size: int = 1000,
        num_blocks: int = 1024,
        block_size: int = 16,
        max_seq_length: int = 4096
    ):
        """
        Initialize the LLM.
        
        Args:
            model_name: Name of the model to load
            device: Device to run inference on ("auto", "mps", "cuda", "cpu")
            dtype: Data type for model weights (None for auto-detection)
            max_queue_size: Maximum number of requests in queue
            num_blocks: Number of memory blocks for KV cache
            block_size: Size of each memory block
            max_seq_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        
        # Initialize the LLM engine
        self.engine = LLMEngine(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_queue_size=max_queue_size,
            num_blocks=num_blocks,
            block_size=block_size,
            max_seq_length=max_seq_length
        )
        
        logger.info(f"LLM initialized with {model_name} on {self.engine.model_runner.device}")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> List[Dict[str, Any]]:
        """
        Generate text for given prompts.
        
        This is the main interface method that follows the sequence diagram:
        User -> LLM.generate() -> LLMEngine.generate() -> Output
        
        Args:
            prompts: Single prompt or list of prompts
            sampling_params: Sampling parameters for generation
            priority: Request priority
            
        Returns:
            List of generation results with the following structure:
            {
                "request_id": int,
                "prompt": str,
                "generated_text": str,
                "tokens_generated": int,
                "total_tokens": int,
                "block_indices": List[int]
            }
        """
        logger.info(f"Generating text for {len(prompts) if isinstance(prompts, list) else 1} prompt(s)")
        
        # Delegate to the engine
        results = self.engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            priority=priority
        )
        
        logger.info(f"Generated {len(results)} result(s)")
        return results
    
    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate text with streaming output.
        
        This method yields tokens as they are generated, providing real-time
        output for better user experience.
        
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
        logger.info(f"Starting streaming generation for prompt: {prompt[:50]}...")
        
        # Delegate to the engine for streaming
        for stream_result in self.engine.generate_stream(
            prompt=prompt,
            sampling_params=sampling_params,
            priority=priority
        ):
            yield stream_result
    
    def generate_single(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> Dict[str, Any]:
        """
        Generate text for a single prompt.
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            priority: Request priority
            
        Returns:
            Single generation result
        """
        results = self.generate([prompt], sampling_params, priority)
        return results[0] if results else {}
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
        system_prompt: str = "You are a helpful AI assistant."
    ) -> Dict[str, Any]:
        """
        Generate a chat response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            sampling_params: Sampling parameters
            system_prompt: System prompt to prepend
            
        Returns:
            Chat response
        """
        # Format messages for the model
        formatted_prompt = self._format_chat_prompt(messages, system_prompt)
        
        return self.generate_single(formatted_prompt, sampling_params)
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
        system_prompt: str = "You are a helpful AI assistant."
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a chat response with streaming output.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            sampling_params: Sampling parameters
            system_prompt: System prompt to prepend
            
        Yields:
            Streaming chat response tokens
        """
        # Format messages for the model
        formatted_prompt = self._format_chat_prompt(messages, system_prompt)
        
        # Use streaming generation
        for stream_result in self.generate_stream(formatted_prompt, sampling_params):
            yield stream_result
    
    def _format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str
    ) -> str:
        """
        Format chat messages into a single prompt.
        
        Args:
            messages: List of message dictionaries
            system_prompt: System prompt
            
        Returns:
            Formatted prompt string
        """
        # Start with system prompt
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # Add conversation messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Add assistant prefix for response
        prompt += "<|im_start|>assistant\n"
        
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM statistics."""
        return self.engine.get_stats()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.engine.get_model_info()
    
    def clear_stats(self) -> None:
        """Clear all statistics."""
        self.engine.clear_stats()
    
    def shutdown(self) -> None:
        """Shutdown the LLM and free resources."""
        logger.info("Shutting down LLM...")
        self.engine.shutdown()
        logger.info("LLM shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    # Convenience methods for common sampling configurations
    def generate_greedy(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        """Generate text using greedy decoding."""
        sampling_params = SamplingParams.greedy(max_tokens=max_tokens)
        return self.generate(prompts, sampling_params)
    
    def generate_creative(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        """Generate text using creative sampling."""
        sampling_params = SamplingParams.creative(max_tokens=max_tokens)
        return self.generate(prompts, sampling_params)
    
    def generate_balanced(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        """Generate text using balanced sampling."""
        sampling_params = SamplingParams.balanced(max_tokens=max_tokens)
        return self.generate(prompts, sampling_params) 