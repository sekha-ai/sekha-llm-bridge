"""FastAPI entry point for LLM Bridge"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from config import settings
from utils.llm_client import llm_client
from services.embedding_service import embedding_service
from services.summarization_service import summarization_service
from services.entity_extraction_service import entity_extraction_service
from services.importance_scorer import importance_scorer_service
from models.requests import EmbedRequest, SummarizeRequest, ExtractRequest, ScoreRequest
from models.responses import HealthResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
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
# Health Endpoint
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health and Ollama connectivity"""
    ollama_status = await llm_client.health_check()
    
    if ollama_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail="Ollama is not available")
    
    return HealthResponse(
        status="healthy",
        ollama_status=ollama_status,
        models_loaded=ollama_status.get("models_available", [])
    )


# ============================================
# Embedding Endpoints
# ============================================

@app.post("/api/v1/embed", tags=["Embeddings"])
async def generate_embedding(request: EmbedRequest):
    """Generate embedding for text"""
    try:
        return await embedding_service.generate_embedding(request)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Summarization Endpoints
# ============================================

@app.post("/api/v1/summarize", tags=["Summarization"])
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


# ============================================
# Entity Extraction Endpoints
# ============================================

@app.post("/api/v1/extract", tags=["Entity Extraction"])
async def extract_entities(request: ExtractRequest):
    """Extract entities from text"""
    try:
        return await entity_extraction_service.extract_entities(request)
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Importance Scoring Endpoints
# ============================================

@app.post("/api/v1/score", tags=["Importance Scoring"])
async def score_importance(request: ScoreRequest):
    """Score conversation importance"""
    try:
        return await importance_scorer_service.score_importance(request)
    except Exception as e:
        logger.error(f"Importance scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
