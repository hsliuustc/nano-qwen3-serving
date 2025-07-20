"""
OpenAI-compatible service wrapper for the nano Qwen3 serving engine.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from loguru import logger

from ..core.llm import LLM
from ..core.sampling_params import SamplingParams
from .openai_models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatChoice, Usage,
    CompletionRequest, CompletionResponse, CompletionChoice,
    ModelsResponse, HealthResponse, StatsResponse, ModelInfo,
    generate_response_id, get_current_timestamp, create_model_info
)


class OpenAICompatibleService:
    """OpenAI-compatible service wrapper."""
    
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
        """Initialize the service."""
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_queue_size = max_queue_size
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_seq_length = max_seq_length
        
        self.start_time = time.time()
        self.llm: Optional[LLM] = None
        
    async def startup(self):
        """Startup the service."""
        logger.info(f"Starting OpenAI-compatible service with model: {self.model_name}")
        
        # Initialize LLM in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        self.llm = await loop.run_in_executor(
            None,
            lambda: LLM(
                model_name=self.model_name,
                device=self.device,
                dtype=self.dtype,
                max_queue_size=self.max_queue_size,
                num_blocks=self.num_blocks,
                block_size=self.block_size,
                max_seq_length=self.max_seq_length
            )
        )
        
        logger.info("OpenAI-compatible service startup complete")
    
    async def shutdown(self):
        """Shutdown the service."""
        logger.info("Shutting down OpenAI-compatible service")
        
        if self.llm:
            await asyncio.get_event_loop().run_in_executor(None, self.llm.shutdown)
        
        logger.info("OpenAI-compatible service shutdown complete")
    
    def _convert_to_sampling_params(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> SamplingParams:
        """Convert OpenAI request to internal sampling parameters."""
        return SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 100,
            stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None
        )
    
    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Format chat messages as a prompt."""
        # Simple formatting - can be enhanced for better chat handling
        formatted = ""
        for message in messages:
            role = message.role
            content = message.content
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        formatted += "Assistant: "
        return formatted
    
    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate chat completions."""
        try:
            # Validate model
            if request.model not in ["qwen3-0.6b", "Qwen/Qwen3-0.6B"]:
                raise ValueError(f"Model {request.model} not supported")
            
            # Convert to internal format
            sampling_params = self._convert_to_sampling_params(request)
            prompt = self._format_chat_prompt(request.messages)
            
            # Generate response
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.llm.generate_single(prompt, sampling_params)
            )
            
            # Convert to OpenAI format
            generated_text = result.get("generated_text", "")
            tokens_generated = result.get("tokens_generated", 0)
            total_tokens = result.get("total_tokens", 0)
            
            return ChatCompletionResponse(
                id=generate_response_id(),
                object="chat.completion",
                created=get_current_timestamp(),
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=generated_text
                        ),
                        finish_reason="stop" if tokens_generated < (sampling_params.max_tokens or 100) else "length"
                    )
                ],
                usage=Usage(
                    prompt_tokens=total_tokens - tokens_generated,
                    completion_tokens=tokens_generated,
                    total_tokens=total_tokens
                )
            )
            
        except Exception as e:
            logger.error(f"Error in chat_completions: {e}")
            raise
    
    async def chat_completions_stream(self, request: ChatCompletionRequest):
        """Generate streaming chat completions."""
        try:
            # Validate model
            if request.model not in ["qwen3-0.6b", "Qwen/Qwen3-0.6B"]:
                raise ValueError(f"Model {request.model} not supported")
            
            # Convert to internal format
            sampling_params = self._convert_to_sampling_params(request)
            prompt = self._format_chat_prompt(request.messages)
            
            # Generate streaming response
            loop = asyncio.get_event_loop()
            
            async def stream_generator():
                try:
                    # Run streaming in executor
                    for result in self.llm.generate_stream(prompt, sampling_params):
                        # Format as OpenAI streaming response
                        if 'token' in result:
                            stream_data = {
                                "id": generate_response_id(),
                                "object": "chat.completion.chunk",
                                "created": get_current_timestamp(),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": result['token']},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            
                            if result.get('finished', False):
                                stream_data["choices"][0]["finish_reason"] = "stop"
                            
                            yield f"data: {stream_data}\n\n"
                            
                            if result.get('finished', False):
                                yield "data: [DONE]\n\n"
                                break
                            
                            # Small delay to prevent blocking
                            await asyncio.sleep(0.001)
                            
                except Exception as e:
                    logger.error(f"Error in stream generator: {e}")
                    error_data = {
                        "error": {
                            "message": str(e),
                            "type": "server_error"
                        }
                    }
                    yield f"data: {error_data}\n\n"
            
            return stream_generator()
            
        except Exception as e:
            logger.error(f"Error in chat_completions_stream: {e}")
            raise
    
    async def completions(self, request: CompletionRequest) -> CompletionResponse:
        """Generate text completions (legacy)."""
        try:
            # Validate model
            if request.model not in ["qwen3-0.6b", "Qwen/Qwen3-0.6B"]:
                raise ValueError(f"Model {request.model} not supported")
            
            # Convert to internal format
            sampling_params = self._convert_to_sampling_params(request)
            prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
            
            # Generate response
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.llm.generate_single(prompt, sampling_params)
            )
            
            # Convert to OpenAI format
            generated_text = result.get("generated_text", "")
            tokens_generated = result.get("tokens_generated", 0)
            total_tokens = result.get("total_tokens", 0)
            
            return CompletionResponse(
                id=generate_response_id(),
                object="text_completion",
                created=get_current_timestamp(),
                model=request.model,
                choices=[
                    CompletionChoice(
                        text=generated_text,
                        index=0,
                        finish_reason="stop" if tokens_generated < (sampling_params.max_tokens or 100) else "length"
                    )
                ],
                usage=Usage(
                    prompt_tokens=total_tokens - tokens_generated,
                    completion_tokens=tokens_generated,
                    total_tokens=total_tokens
                )
            )
            
        except Exception as e:
            logger.error(f"Error in completions: {e}")
            raise
    
    def get_models(self) -> ModelsResponse:
        """Get available models."""
        return ModelsResponse(
            object="list",
            data=[
                create_model_info("qwen3-0.6b"),
                create_model_info("Qwen/Qwen3-0.6B")
            ]
        )
    
    def get_health(self) -> HealthResponse:
        """Get service health information."""
        try:
            model_info = self.llm.get_model_info() if self.llm else {}
            uptime = time.time() - self.start_time
            
            return HealthResponse(
                status="healthy",
                model_info=model_info,
                uptime=uptime
            )
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            raise
    
    def get_stats(self) -> StatsResponse:
        """Get service statistics."""
        try:
            stats = self.llm.get_stats() if self.llm else {}
            
            return StatsResponse(
                uptime=stats.get("uptime", 0),
                total_requests=stats.get("total_requests", 0),
                completed_requests=stats.get("completed_requests", 0),
                failed_requests=stats.get("failed_requests", 0),
                success_rate=stats.get("success_rate", 0),
                requests_per_second=stats.get("requests_per_second", 0),
                memory_stats=stats.get("memory_stats", {}),
                model_stats=stats.get("model_stats", {})
            )
        except Exception as e:
            logger.error(f"Error in stats: {e}")
            raise


# Global service instance
service: Optional[OpenAICompatibleService] = None


@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager."""
    global service
    
    # Startup
    service = OpenAICompatibleService()
    await service.startup()
    
    yield
    
    # Shutdown
    if service:
        await service.shutdown() 