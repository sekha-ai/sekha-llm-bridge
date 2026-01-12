"""FastAPI entry point for LLM Bridge"""

# src/sekha_llm_bridge/main.py

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

# Internal imports - all from the package
from sekha_llm_bridge.config import settings
from sekha_llm_bridge.models import (
    EmbedRequest, EmbedResponse,
    SummarizeRequest, SummarizeResponse,
    ExtractRequest, ExtractResponse,
    ScoreRequest, ScoreResponse,
    HealthResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionChoice, ChatCompletionUsage
)
from sekha_llm_bridge.utils.llm_client import llm_client
from sekha_llm_bridge.services.embedding_service import embedding_service
from sekha_llm_bridge.services.summarization_service import summarization_service
from sekha_llm_bridge.services.entity_extraction_service import entity_extraction_service
from sekha_llm_bridge.services.importance_scorer import importance_scorer_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Starting Sekha LLM Bridge")
    
    # Check Ollama health on startup
    health = await llm_client.health_check()
    if health["status"] == "healthy":
        logger.info(f"✅ Ollama is healthy: {health['models_available']}")
    else:
        logger.warning(f"⚠️ Ollama health check failed: {health.get('reason')}")
    
    yield
    
    logger.info("👋 Shutting down Sekha LLM Bridge")


# Create FastAPI app
app = FastAPI(
    title="Sekha LLM Bridge",
    description="LLM operations service for Project Sekha",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ============================================
# Health Endpoint (with timestamp)
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health and Ollama connectivity"""
    ollama_status = await llm_client.health_check()
    
    if ollama_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail="Ollama is not available")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        ollama_status=ollama_status,
        models_loaded=ollama_status.get("models_available", [])
    )


# ============================================
# Root-level endpoints (for backward compatibility/tests)
# ============================================

@app.post("/embed", response_model=EmbedResponse, tags=["Embeddings"])
async def embed_text_root(request: EmbedRequest):
    """Root-level embedding endpoint"""
    try:
        embedding = await llm_client.generate_embedding(request.text, request.model)
        return EmbedResponse(
            embedding=embedding,
            model=request.model or settings.embedding_model,
            dimension=len(embedding)
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
            detail=f"Invalid level: {request.level}. Must be daily, weekly, or monthly"
        )
    
    try:
        # Build prompt from messages
        messages_str = "\n".join(request.messages)
        prompt = f"Summarize these messages. Level: {request.level}\n\n{messages_str}"
        
        summary = await llm_client.generate_completion(
            [{"role": "user", "content": prompt}],
            request.model
        )
        
        return SummarizeResponse(
            summary=summary.strip(),
            level=request.level,
            model=request.model or settings.summarization_model,
            message_count=len(request.messages)
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
@app.post("/score_importance", response_model=ScoreResponse, tags=["Importance Scoring"])
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
            detail=f"Invalid level: {request.level}. Must be daily, weekly, or monthly"
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
        # Convert ChatMessage objects to dicts for llm_client
        messages_dict = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Generate completion via llm_client
        content = await llm_client.generate_completion(
            messages=messages_dict,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 2000
        )
        
        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model or settings.default_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": content},
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=sum(len(m.content.split()) for m in request.messages),
                completion_tokens=len(content.split()),
                total_tokens=sum(len(m.content.split()) for m in request.messages) + len(content.split())
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Chat"])
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
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )