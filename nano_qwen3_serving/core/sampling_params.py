"""
Sampling parameters for text generation.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator, model_validator
from loguru import logger


class SamplingParams(BaseModel):
    """
    Configuration parameters for text generation sampling.
    
    This class defines all the parameters that control how the model generates text,
    including temperature, top-p, top-k, and other sampling strategies.
    """
    
    # Core sampling parameters
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    top_k: int = Field(default=-1, ge=-1, description="Top-k sampling parameter (-1 for no limit)")
    
    # Generation control
    max_tokens: int = Field(default=100, ge=1, description="Maximum number of tokens to generate")
    min_tokens: int = Field(default=0, ge=0, description="Minimum number of tokens to generate")
    
    # Stopping criteria
    stop_sequences: Optional[List[str]] = Field(default=None, description="Sequences that stop generation")
    stop_token_ids: Optional[List[int]] = Field(default=None, description="Token IDs that stop generation")
    
    # Repetition control
    repetition_penalty: float = Field(default=1.0, ge=0.0, description="Penalty for repetition")
    length_penalty: float = Field(default=1.0, description="Penalty for sequence length")
    
    # Advanced parameters
    do_sample: bool = Field(default=True, description="Whether to use sampling vs greedy decoding")
    use_beam_search: bool = Field(default=False, description="Whether to use beam search")
    num_beams: int = Field(default=1, ge=1, description="Number of beams for beam search")
    
    # MPS-specific optimizations
    use_cache: bool = Field(default=True, description="Whether to use KV cache")
    cache_precision: str = Field(default="float16", description="Precision for KV cache")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v == 0.0:
            logger.warning("Temperature is 0.0, which will result in deterministic (greedy) generation")
        return v
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v == 0:
            raise ValueError("top_k must be -1 (no limit) or >= 1")
        return v
    
    @model_validator(mode='after')
    def validate_token_limits(self):
        """Validate that min_tokens is not greater than max_tokens."""
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot be greater than max_tokens")
        return self
    
    @validator('use_beam_search', 'do_sample')
    def validate_sampling_strategy(cls, v, values):
        if 'use_beam_search' in values and 'do_sample' in values:
            if values['use_beam_search'] and values['do_sample']:
                logger.warning("Beam search is enabled, do_sample will be ignored")
        return v
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary format."""
        return self.model_dump()
    
    def __str__(self) -> str:
        """String representation of sampling parameters."""
        params = []
        if self.temperature != 1.0:
            params.append(f"temp={self.temperature}")
        if self.top_p != 1.0:
            params.append(f"top_p={self.top_p}")
        if self.top_k != -1:
            params.append(f"top_k={self.top_k}")
        if self.max_tokens != 512:
            params.append(f"max_tokens={self.max_tokens}")
        
        return f"SamplingParams({', '.join(params)})"
    
    @classmethod
    def greedy(cls, max_tokens: int = 100) -> "SamplingParams":
        """Create parameters for greedy decoding."""
        return cls(
            temperature=0.0,
            do_sample=False,
            use_beam_search=False,
            max_tokens=max_tokens
        )
    
    @classmethod
    def creative(cls, max_tokens: int = 100) -> "SamplingParams":
        """Create parameters for creative text generation."""
        return cls(
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            max_tokens=max_tokens
        )
    
    @classmethod
    def balanced(cls, max_tokens: int = 100) -> "SamplingParams":
        """Create parameters for balanced text generation."""
        return cls(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=max_tokens
        ) 