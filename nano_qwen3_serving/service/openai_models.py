"""
OpenAI-compatible API models for the nano Qwen3 serving service.
"""

import time
import uuid
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    total_tokens: int = Field(..., description="Total number of tokens")


class ChatChoice(BaseModel):
    """Chat completion choice."""
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: str = Field(..., description="Reason for finishing")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    n: Optional[int] = Field(default=1, ge=1, le=1, description="Number of completions (always 1)")
    stream: Optional[bool] = Field(default=False, description="Enable streaming")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(default=None, description="Logit bias")
    user: Optional[str] = Field(default=None, description="User identifier")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatChoice] = Field(..., description="Generated choices")
    usage: Usage = Field(..., description="Token usage")


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request (legacy)."""
    model: str = Field(..., description="Model name")
    prompt: Union[str, List[str]] = Field(..., description="Input prompt")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    n: Optional[int] = Field(default=1, ge=1, le=1, description="Number of completions")
    stream: Optional[bool] = Field(default=False, description="Enable streaming")
    max_tokens: Optional[int] = Field(default=16, ge=1, description="Maximum tokens to generate")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(default=None, description="Logit bias")
    user: Optional[str] = Field(default=None, description="User identifier")


class CompletionChoice(BaseModel):
    """Completion choice."""
    text: str = Field(..., description="Generated text")
    index: int = Field(..., description="Choice index")
    logprobs: Optional[Dict[str, Any]] = Field(default=None, description="Log probabilities")
    finish_reason: str = Field(..., description="Reason for finishing")


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="text_completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[CompletionChoice] = Field(..., description="Generated choices")
    usage: Usage = Field(..., description="Token usage")


class ModelInfo(BaseModel):
    """Model information."""
    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(..., description="Model owner")


class ModelsResponse(BaseModel):
    """Models list response."""
    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of models")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    uptime: float = Field(..., description="Service uptime in seconds")


class StatsResponse(BaseModel):
    """Statistics response."""
    uptime: float = Field(..., description="Service uptime")
    total_requests: int = Field(..., description="Total requests processed")
    completed_requests: int = Field(..., description="Completed requests")
    failed_requests: int = Field(..., description="Failed requests")
    success_rate: float = Field(..., description="Success rate")
    requests_per_second: float = Field(..., description="Requests per second")
    memory_stats: Dict[str, Any] = Field(..., description="Memory statistics")
    model_stats: Dict[str, Any] = Field(..., description="Model statistics")


class ErrorResponse(BaseModel):
    """Error response."""
    error: Dict[str, Any] = Field(..., description="Error information")


# Utility functions
def generate_response_id() -> str:
    """Generate a unique response ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def get_current_timestamp() -> int:
    """Get current timestamp."""
    return int(time.time())


def create_model_info(model_id: str, owned_by: str = "nano-qwen3-serving") -> ModelInfo:
    """Create model information."""
    return ModelInfo(
        id=model_id,
        object="model",
        created=get_current_timestamp(),
        owned_by=owned_by
    ) 