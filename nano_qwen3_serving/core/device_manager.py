"""
Device manager for multi-backend support (MPS, CUDA, CPU).
"""

import torch
from typing import Dict, Any, Optional, Tuple
from loguru import logger


class DeviceManager:
    """
    Manages device detection, configuration, and optimization for different backends.
    
    Supports:
    - MPS (Apple Silicon)
    - CUDA (NVIDIA GPUs)
    - CPU (fallback)
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize device manager.
        
        Args:
            device: Device to use ("auto", "mps", "cuda", "cpu")
        """
        self.device = self._detect_device(device)
        self.device_info = self._get_device_info()
        self.optimization_config = self._get_optimization_config()
        
        logger.info(f"Device manager initialized: {self.device}")
        logger.info(f"Device info: {self.device_info}")
    
    def _detect_device(self, device: str) -> str:
        """
        Detect and validate the best available device.
        
        Args:
            device: Requested device
            
        Returns:
            Validated device string
        """
        if device == "auto":
            # Auto-detect best available device
            if self._is_cuda_available():
                return "cuda"
            elif self._is_mps_available():
                return "mps"
            else:
                return "cpu"
        
        # Validate requested device
        if device == "cuda":
            if self._is_cuda_available():
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        elif device == "mps":
            if self._is_mps_available():
                return "mps"
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
        elif device == "cpu":
                return "cpu"
        else:
            logger.warning(f"Unknown device '{device}', falling back to CPU")
            return "cpu"
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception:
            return False
    
    def _is_mps_available(self) -> bool:
        """Check if MPS is available."""
        try:
            return torch.backends.mps.is_available()
        except Exception:
            return False
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            "device": self.device,
            "device_type": self._get_device_type(),
            "memory_info": self._get_memory_info(),
            "compute_capability": self._get_compute_capability(),
        }
        
        if self.device == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "current_gpu": torch.cuda.current_device(),
            })
        elif self.device == "mps":
            info.update({
                "gpu_name": "Apple Silicon GPU",
                "mps_available": True,
            })
        
        return info
    
    def _get_device_type(self) -> str:
        """Get device type."""
        if self.device == "cuda":
            return "nvidia_gpu"
        elif self.device == "mps":
            return "apple_gpu"
        else:
            return "cpu"
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for the device."""
        if self.device == "cuda":
            return {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(0),
                "cached": torch.cuda.memory_reserved(0),
            }
        elif self.device == "mps":
            # MPS doesn't provide detailed memory info
            return {
                "total": None,
                "allocated": None,
                "cached": None,
            }
        else:
            # CPU memory info
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
            }
    
    def _get_compute_capability(self) -> Optional[Tuple[int, int]]:
        """Get compute capability (CUDA only)."""
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            return (props.major, props.minor)
        return None
    
    def _get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration for the device."""
        config = {
            "enable_compilation": False,  # 禁用编译，避免kernel问题
            "enable_memory_efficient_attention": False,  # 禁用内存优化attention
            "enable_flash_attention": False,
            "memory_format": "contiguous",  # 使用连续内存格式
            "dtype": torch.float16,
        }
        
        if self.device == "cuda":
            config.update({
                "enable_flash_attention": False,  # 禁用Flash Attention
                "memory_format": "contiguous",  # 使用连续内存格式
                "cuda_memory_fraction": 0.9,
                "enable_tensor_cores": False,  # 禁用Tensor Cores
            })
        elif self.device == "mps":
            config.update({
                "mps_memory_fraction": 0.9,
                "enable_half_precision": True,
            })
        else:  # CPU
            config.update({
                "enable_compilation": False,  # CPU compilation can be slow
                "enable_memory_efficient_attention": False,
                "dtype": torch.float32,  # CPU benefits from float32
                "num_threads": torch.get_num_threads(),
            })
        
        return config
    
    def get_device(self) -> torch.device:
        """Get PyTorch device object."""
        return torch.device(self.device)
    
    def get_dtype(self) -> torch.dtype:
        """Get optimal data type for the device."""
        return self.optimization_config["dtype"]
    
    def optimize_for_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply device-specific optimizations to the model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        logger.info(f"Applying {self.device}-specific optimizations...")
        
        # Move model to device
        model = model.to(self.get_device())
        
        # Apply device-specific optimizations
        if self.device == "cuda":
            model = self._optimize_for_cuda(model)
        elif self.device == "mps":
            model = self._optimize_for_mps(model)
        else:  # CPU
            model = self._optimize_for_cpu(model)
        
        return model
    
    def _optimize_for_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply CUDA-specific optimizations."""
        # Set memory fraction
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(
                self.optimization_config["cuda_memory_fraction"]
            )
        
        # Use channels_last memory format (disabled for compatibility)
        if self.optimization_config["memory_format"] == "channels_last":
            try:
                model = model.to(memory_format=torch.channels_last)
                logger.info("Applied channels_last memory format")
            except Exception as e:
                logger.warning(f"Could not apply channels_last: {e}")
        
        # Enable tensor cores for float16 (disabled for compatibility)
        if self.optimization_config["enable_tensor_cores"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled tensor cores")
        
        # 使用连续内存格式确保兼容性
        try:
            model = model.to(memory_format=torch.contiguous_format)
            logger.info("Applied contiguous memory format for compatibility")
        except Exception as e:
            logger.warning(f"Could not apply contiguous format: {e}")
        
        return model
    
    def _optimize_for_mps(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply MPS-specific optimizations."""
        # Set memory fraction
        if hasattr(torch.mps, 'set_per_process_memory_fraction'):
            torch.mps.set_per_process_memory_fraction(
                self.optimization_config["mps_memory_fraction"]
            )
        
        # Use half precision
        if self.optimization_config["enable_half_precision"]:
            model = model.half()
            logger.info("Applied half precision")
        
        return model
    
    def _optimize_for_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply CPU-specific optimizations."""
        # Set number of threads
        if "num_threads" in self.optimization_config:
            torch.set_num_threads(self.optimization_config["num_threads"])
            logger.info(f"Set CPU threads to {self.optimization_config['num_threads']}")
        
        # Use float32 for CPU (better performance)
        model = model.float()
        logger.info("Applied float32 precision for CPU")
        
        return model
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {
            "device": self.device,
            "device_info": self.device_info,
        }
        
        if self.device == "cuda":
            stats.update({
                "allocated": torch.cuda.memory_allocated(0),
                "cached": torch.cuda.memory_reserved(0),
                "max_allocated": torch.cuda.max_memory_allocated(0),
            })
        elif self.device == "mps":
            # MPS doesn't provide detailed memory stats
            stats.update({
                "allocated": None,
                "cached": None,
                "max_allocated": None,
            })
        else:  # CPU
            import psutil
            memory = psutil.virtual_memory()
            stats.update({
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
            })
        
        return stats
    
    def clear_memory(self) -> None:
        """Clear device memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("Cleared CUDA memory cache")
        elif self.device == "mps":
            # MPS doesn't have explicit memory clearing
            logger.info("MPS memory clearing not available")
        else:
            # CPU memory is managed by the OS
            logger.info("CPU memory clearing not needed")
    
    def __str__(self) -> str:
        return f"DeviceManager(device={self.device}, info={self.device_info})" 