"""
Model execution and inference on multiple backends (MPS, CUDA, CPU).
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from loguru import logger
import time
from .device_manager import DeviceManager


class ModelRunner:
    """
    Handles model execution and inference on multiple backends.
    
    This class manages the actual forward passes through the model,
    including attention computation, KV cache management, and sampling.
    Supports MPS (Apple Silicon), CUDA (NVIDIA), and CPU backends.
    """
    
    def __init__(
        self,
        model_name: str = "/zx_data1/nano-vllm/models/Qwen3-0.6B",  # "Qwen/Qwen3-0.6B",
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
        max_seq_length: int = 4096
    ):
        """
        Initialize the model runner.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to run inference on ("auto", "mps", "cuda", "cpu")
            dtype: Data type for model weights (None for auto-detection)
            use_cache: Whether to use KV cache
            max_seq_length: Maximum sequence length
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.max_seq_length = max_seq_length
        
        # Initialize device manager
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        
        # Set dtype (auto-detect if not provided)
        if dtype is None:
            self.dtype = self.device_manager.get_dtype()
        else:
            self.dtype = dtype
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # KV cache
        self.kv_cache = None
        self.cache_enabled = use_cache
        
        # Performance tracking
        self.inference_times = []
        self.tokens_generated = 0
        
        logger.info(f"ModelRunner initialized with {model_name} on {self.device}")
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Apply device-specific optimizations
            if self.model is not None:
                self.model = self.device_manager.optimize_for_device(self.model)
            
            logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def run_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Run a forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Previous key-value pairs for caching
            use_cache: Whether to use KV cache
            
        Returns:
            Model outputs including logits and new key-value pairs
        """
        if use_cache is None:
            use_cache = self.cache_enabled
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True
                )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Update statistics
            self.tokens_generated += input_ids.shape[1]
            
            logger.debug(f"Model inference completed in {inference_time:.4f}s")
            
            return {
                "logits": outputs.logits,
                "past_key_values": outputs.past_key_values,
                "inference_time": inference_time
            }
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise
    
    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        sampling_params: Any,  # SamplingParams
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Generate the next token given input and sampling parameters.
        
        Args:
            input_ids: Input token IDs
            sampling_params: Sampling parameters
            past_key_values: Previous key-value pairs
            
        Returns:
            Tuple of (next_token_id, new_past_key_values)
        """
        # Create attention mask
        # When using KV cache, the attention mask should account for the full sequence
        if past_key_values is not None:
            # We have cached tokens, so create a mask for the full sequence
            # The input_ids contains only the new token, but we need to mask for the full sequence
            total_length = past_key_values[0][0].shape[2] + input_ids.shape[1]  # cached + new
            attention_mask = torch.ones((input_ids.shape[0], total_length), device=input_ids.device, dtype=input_ids.dtype)
        else:
            # No cache, use simple mask
            attention_mask = torch.ones_like(input_ids)
        
        # Run model forward pass
        outputs = self.run_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=sampling_params.use_cache
        )
        
        logits = outputs["logits"]
        new_past_key_values = outputs["past_key_values"]
        
        # Get logits for the last token
        next_token_logits = logits[:, -1, :]
        
        # Apply sampling
        next_token_id = self._sample_token(next_token_logits, sampling_params)
        
        return next_token_id, new_past_key_values
    
    def _sample_token(self, logits: torch.Tensor, sampling_params: Any) -> torch.Tensor:
        """
        Sample the next token based on logits and sampling parameters.
        
        Args:
            logits: Logits for the next token
            sampling_params: Sampling parameters
            
        Returns:
            Sampled token ID
        """
        # Apply temperature
        if sampling_params.temperature != 1.0:
            if sampling_params.temperature == 0.0:
                # For greedy decoding, don't divide by zero
                # Just use the original logits
                pass
            else:
                logits = logits / sampling_params.temperature
        
        # Apply top-k filtering
        if sampling_params.top_k > 0:
            top_k = min(sampling_params.top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Apply top-p (nucleus) sampling
        if sampling_params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > sampling_params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        if sampling_params.do_sample and sampling_params.temperature > 0.0:
            # Handle potential infinite values
            logits = torch.clamp(logits, min=-1e10, max=1e10)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token
    
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Token IDs tensor
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        return tokens.to(self.device)
    
    def detokenize(self, token_ids: torch.Tensor) -> str:
        """
        Detokenize token IDs to text.
        
        Args:
            token_ids: Token IDs to detokenize
            
        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "num_parameters": self.model.num_parameters(),
            "max_seq_length": self.max_seq_length,
            "use_cache": self.cache_enabled
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.inference_times:
            return {
                "average_inference_time": 0.0,
                "total_inference_time": 0.0,
                "tokens_generated": self.tokens_generated,
                "inference_count": 0,
                "tokens_per_second": 0.0,
                "device_stats": self.device_manager.get_memory_stats()
            }
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        total_time = sum(self.inference_times)
        
        return {
            "average_inference_time": avg_time,
            "total_inference_time": total_time,
            "tokens_generated": self.tokens_generated,
            "inference_count": len(self.inference_times),
            "tokens_per_second": self.tokens_generated / total_time if total_time > 0 else 0.0,
            "device_stats": self.device_manager.get_memory_stats()
        }
    
    def clear_performance_stats(self) -> None:
        """Clear performance statistics."""
        self.inference_times.clear()
        self.tokens_generated = 0
        logger.info("Performance statistics cleared")
    
    def optimize_for_inference(self) -> None:
        """Apply optimizations for inference performance."""
        if self.model is None:
            return
        
        logger.info("Applying inference optimizations...")
        
        # 1. Enable torch.compile for faster inference (PyTorch 2.0+)
        # Disabled due to CUDA Graph compatibility issues
        # try:
        #     if hasattr(torch, 'compile'):
        #         self.model = torch.compile(self.model, mode="reduce-overhead")
        #         logger.info("Applied torch.compile optimization")
        # except Exception as e:
        #     logger.warning(f"torch.compile not available: {e}")
        logger.info("torch.compile disabled for CUDA Graph compatibility")
        
        # 2. Enable memory efficient attention if available
        try:
            if hasattr(self.model.config, 'use_memory_efficient_attention'):
                self.model.config.use_memory_efficient_attention = True
                logger.info("Enabled memory efficient attention")
        except Exception as e:
            logger.warning(f"Memory efficient attention not available: {e}")
        
        # 3. Set model to inference mode
        self.model.eval()
        
        # 4. Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # 5. Device-specific optimizations are handled by DeviceManager
        # Additional optimizations can be applied here if needed
    
    def _optimize_for_mps(self) -> None:
        """Apply MPS-specific optimizations (legacy method)."""
        # This method is kept for backward compatibility
        # Device-specific optimizations are now handled by DeviceManager
        logger.info("MPS optimizations handled by DeviceManager")
    
    def enable_fast_inference(self) -> None:
        """Enable fast inference mode with aggressive optimizations."""
        if self.model is None:
            return
        
        logger.info("Enabling fast inference mode...")
        
        # 1. Use torch.inference_mode() for maximum performance
        self.inference_mode = torch.inference_mode()
        
        # 2. Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 3. Use channels_last memory format for better performance
        try:
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("Applied channels_last memory format")
        except Exception as e:
            logger.warning(f"Could not apply channels_last: {e}")
        
        # 4. Enable flash attention if available
        try:
            if hasattr(self.model.config, 'use_flash_attention_2'):
                self.model.config.use_flash_attention_2 = True
                logger.info("Enabled Flash Attention 2")
        except Exception as e:
            logger.warning(f"Flash Attention 2 not available: {e}")
    
    def run_model_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Run batched forward pass through the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask
            past_key_values: Previous key-value pairs for caching
            use_cache: Whether to use KV cache
            
        Returns:
            Model outputs including logits and new key-value pairs
        """
        if use_cache is None:
            use_cache = self.cache_enabled
        
        start_time = time.time()
        
        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True
                )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Update statistics
            self.tokens_generated += input_ids.shape[1]
            
            logger.debug(f"Batch inference completed in {inference_time:.4f}s (batch_size={input_ids.shape[0]})")
            
            return {
                "logits": outputs.logits,
                "past_key_values": outputs.past_key_values,
                "inference_time": inference_time
            }
            
        except Exception as e:
            logger.error(f"Batch model inference failed: {e}")
            raise 