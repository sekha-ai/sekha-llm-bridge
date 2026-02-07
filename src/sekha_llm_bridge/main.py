"""FastAPI entry point for LLM Bridge"""

# src/sekha_llm_bridge/main.py

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

# Internal imports - all from the package
from sekha_llm_bridge.config import get_settings
from sekha_llm_bridge.models import (ChatCompletionChoice,
                                     ChatCompletionRequest,
                                     ChatCompletionResponse,
                                     ChatCompletionUsage, EmbedRequest,
                                     EmbedResponse, ExtractRequest,
                                     ExtractResponse, HealthResponse,
                                     ScoreRequest, ScoreResponse,
                                     SummarizeRequest, SummarizeResponse)
from sekha_llm_bridge.registry import registry
# V2.0 routing (multi-provider support)
from sekha_llm_bridge.routes_v2 import router as v2_router
from sekha_llm_bridge.services.embedding_service import embedding_service
from sekha_llm_bridge.services.entity_extraction_service import \
    entity_extraction_service
from sekha_llm_bridge.services.importance_scorer import \
    importance_scorer_service
from sekha_llm_bridge.services.summarization_service import \
    summarization_service
from sekha_llm_bridge.utils.llm_client import llm_client

# Configure logging with fallback
try:
    settings = get_settings()
    log_level = settings.log_level.upper()
except RuntimeError:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Starting Sekha LLM Bridge v2.0")

    # Check provider health on startup
    try:
        health = registry.get_provider_health()
        healthy_count = sum(
            1
            for p in health.values()
            if p.get("circuit_breaker", {}).get("state") == "closed"
        )
        total_providers = len(health)
        logger.info(f"✅ {healthy_count}/{total_providers} providers healthy")

        # List available models
        models = registry.list_all_models()
        logger.info(f"📦 {len(models)} models available")

    except Exception as e:
        logger.error(f"⚠️ Provider health check failed: {e}")

    # Legacy Ollama health check (backward compatibility)
    try:
        health = await llm_client.health_check()
        if health["status"] == "healthy":
            logger.info(
                f"✅ Legacy Ollama client is healthy: {health['models_available']}"
            )
        else:
            logger.warning(
                f"⚠️ Legacy Ollama health check failed: {health.get('reason')}"
            )
    except Exception as e:
        logger.warning(f"Legacy Ollama client not available: {e}")

    yield

    logger.info("👋 Shutting down Sekha LLM Bridge")


# Create FastAPI app
app = FastAPI(
    title="Sekha LLM Bridge",
    description="Multi-provider LLM operations service for Project Sekha",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include V2.0 routing endpoints
app.include_router(v2_router)

# Add Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ============================================
# Health Endpoint (with timestamp)
# ============================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health and provider connectivity"""
    try:
        # Check v2.0 providers
        provider_health = registry.get_provider_health()
        healthy_count = sum(
            1
            for p in provider_health.values()
            if p.get("circuit_breaker", {}).get("state") == "closed"
        )

        # Also check legacy Ollama
        ollama_status = await llm_client.health_check()

        return HealthResponse(
            status="healthy" if healthy_count > 0 else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            ollama_status=ollama_status,
            models_loaded=registry.list_all_models(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


# ============================================
# Root-level endpoints (for backward compatibility/tests)
# ============================================


@app.post("/embed", response_model=EmbedResponse, tags=["Embeddings"])
async def embed_text_root(request: EmbedRequest):
    """Root-level embedding endpoint"""
    try:
        settings = get_settings()
        embedding = await llm_client.generate_embedding(request.text, request.model)
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        tokens_used = len(request.text) // 4
        default_model = settings.default_models.embedding or "nomic-embed-text"
        return EmbedResponse(
            embedding=embedding,
            model=request.model or default_model,
            dimension=len(embedding),
            tokens_used=tokens_used,
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize_messages_root(request: SummarizeRequest):
    """Root-level summarization endpoint"""
    if request.level not in ["daily", "weekly", "monthly"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level: {request.level}. Must be daily, weekly, or monthly",
        )

    try:
        settings = get_settings()
        # Build prompt from messages
        messages_str = "\n".join(request.messages)
        prompt = f"Summarize these messages. Level: {request.level}\n\n{messages_str}"

        summary = await llm_client.generate_completion(
            [{"role": "user", "content": prompt}], request.model
        )

        # Estimate tokens
        tokens_used = (len(messages_str) + len(summary)) // 4
        default_model = settings.default_models.chat_smart or "llama3.1:8b"

        return SummarizeResponse(
            summary=summary.strip(),
            level=request.level,
            model=request.model or default_model,
            message_count=len(request.messages),
            tokens_used=tokens_used,
        )
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings", response_model=EmbedResponse, tags=["Embeddings"])
async def embeddings_alias(request: EmbedRequest):
    """Alias for /embed (some tests expect this path)"""
    return await embed_text_root(request)


@app.post("/extract", response_model=ExtractResponse, tags=["Entity Extraction"])
async def extract_entities_root(request: ExtractRequest):
    """Root-level entity extraction endpoint"""
    try:
        return await entity_extraction_service.extract_entities(request)
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=ScoreResponse, tags=["Importance Scoring"])
async def score_importance_root(request: ScoreRequest):
    """Root-level importance scoring endpoint"""
    try:
        return await importance_scorer_service.score_importance(request)
    except Exception as e:
        logger.error(f"Importance scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Alias for /score_importance (some tests/integrations expect this path)
@app.post(
    "/score_importance", response_model=ScoreResponse, tags=["Importance Scoring"]
)
async def score_importance_alias(request: ScoreRequest):
    """Alias endpoint for importance scoring (compatibility)"""
    return await score_importance_root(request)


# ============================================
# API v1 endpoints (primary)
# ============================================


@app.post("/api/v1/embed", response_model=EmbedResponse, tags=["Embeddings"])
async def generate_embedding(request: EmbedRequest):
    """Generate embedding for text"""
    try:
        return await embedding_service.generate_embedding(request)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def generate_summary(request: SummarizeRequest):
    """Generate summary from messages"""
    if request.level not in ["daily", "weekly", "monthly"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level: {request.level}. Must be daily, weekly, or monthly",
        )

    try:
        return await summarization_service.generate_summary(request)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/extract", response_model=ExtractResponse, tags=["Entity Extraction"])
async def extract_entities_v1(request: ExtractRequest):
    """Extract entities from text"""
    try:
        return await entity_extraction_service.extract_entities(request)
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/score", response_model=ScoreResponse, tags=["Importance Scoring"])
async def score_importance_v1(request: ScoreRequest):
    """Score conversation importance"""
    try:
        return await importance_scorer_service.score_importance(request)
    except Exception as e:
        logger.error(f"Importance scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Chat Completions Endpoints (OpenAI-compatible)
# ============================================


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Chat"])
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Routes chat requests through bridge to Ollama.
    """
    try:
        settings = get_settings()
        # Convert ChatMessage objects to dicts for llm_client
        messages_dict = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Generate completion via llm_client
        content = await llm_client.generate_completion(
            messages=messages_dict,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 2000,
        )

        # Build OpenAI-compatible response
        default_model = settings.default_models.chat_smart or "llama3.1:8b"
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model or default_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": content},
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=sum(len(m.content.split()) for m in request.messages),
                completion_tokens=len(content.split()),
                total_tokens=sum(len(m.content.split()) for m in request.messages)
                + len(content.split()),
            ),
        )

        return response

    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Chat"]
)
async def chat_completions_v1(request: ChatCompletionRequest):
    """API v1 version of chat completions"""
    return await chat_completions(request)


# ============================================
# Root Endpoint
# ============================================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Sekha LLM Bridge",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Multi-provider support",
            "Automatic routing",
            "Cost estimation",
            "Circuit breakers",
            "Provider fallback",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    try:
        settings = get_settings()
        host = settings.server_host
        port = settings.server_port
        log_level_str = settings.log_level.lower()
    except RuntimeError:
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "5001"))
        log_level_str = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level_str,
    )
