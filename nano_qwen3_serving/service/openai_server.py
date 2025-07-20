"""
FastAPI server with OpenAI-compatible API endpoints.
"""

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from .openai_service import OpenAICompatibleService, lifespan
from .openai_models import (
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    ModelsResponse, HealthResponse, StatsResponse, ErrorResponse
)


# Create FastAPI app
app = FastAPI(
    title="Nano Qwen3 OpenAI-Compatible API",
    description="OpenAI-compatible API for the nano Qwen3 serving engine",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Nano Qwen3 OpenAI-Compatible API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/v1/models"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    from .openai_service import service
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return service.get_health()


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Statistics endpoint."""
    from .openai_service import service
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return service.get_stats()


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    from .openai_service import service
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return service.get_models()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Generate chat completions."""
    from .openai_service import service
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                service.chat_completions_stream(request),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
        else:
            # Return regular response
            return await service.chat_completions(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """Generate text completions (legacy)."""
    from .openai_service import service
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if request.stream:
            # For now, return error for streaming completions
            raise HTTPException(status_code=400, detail="Streaming not supported for completions endpoint")
        else:
            return await service.completions(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return ErrorResponse(
        error={
            "message": "Endpoint not found",
            "type": "not_found",
            "code": 404
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return ErrorResponse(
        error={
            "message": "Internal server error",
            "type": "server_error",
            "code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 