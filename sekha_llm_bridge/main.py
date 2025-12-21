from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import litellm

from .config import settings
from .schemas import (
    EmbedRequest,
    EmbedResponse,
    SummarizeRequest,
    SummarizeResponse,
    ExtractRequest,
    ExtractResponse,
    ExtractedEntity,
    ScoreRequest,
    ScoreResponse,
)
from .tasks import (
    embed_text_task,
    summarize_messages_task,
    extract_entities_task,
    score_importance_task,
)


app = FastAPI(title="Sekha LLM Bridge", version="0.1.0")

# Basic CORS (optional; can be tightened later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def init_litellm() -> None:
    # Configure LiteLLM global settings if needed
    # e.g., litellm.set_verbose, api_base, etc.
    pass


# -------- Health Check --------

@app.get("/health")
async def health():
    # Check Redis / Celery
    redis_ok = False
    try:
        from redis import Redis

        r = Redis.from_url(settings.redis_url)
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    # Check Ollama
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Simple health-like ping: list models or root
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status": "ok" if redis_ok and ollama_ok else "degraded",
        "redis": redis_ok,
        "ollama": ollama_ok,
    }


# -------- Embed --------

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    try:
        # For low latency, you can call litellm directly here
        # or push to Celery for heavy workloads.
        embedding = embed_text_task.delay(req.text, req.model).get(timeout=30)
        model_name = req.model or settings.default_embed_model
        return EmbedResponse(embedding=embedding, model=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- Summarize --------

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    if req.level not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="level must be daily, weekly, or monthly")

    try:
        result = summarize_messages_task.delay(req.messages, req.level, req.model).get(timeout=120)
        model_name = req.model or settings.default_summarize_model
        return SummarizeResponse(summary=result, model=model_name, level=req.level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- Extract Entities --------

@app.post("/extract_entities", response_model=ExtractResponse)
async def extract_entities(req: ExtractRequest):
    try:
        raw_entities = extract_entities_task.delay(req.text, req.model).get(timeout=60)
        entities = [
            ExtractedEntity(
                type=e.get("type", "UNKNOWN"),
                value=e.get("value", ""),
                confidence=float(e.get("confidence", 0.5)),
            )
            for e in raw_entities
        ]
        model_name = req.model or settings.default_summarize_model
        return ExtractResponse(entities=entities, model=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- Score Importance --------

@app.post("/score_importance", response_model=ScoreResponse)
async def score_importance(req: ScoreRequest):
    try:
        score = score_importance_task.delay(req.text, req.model).get(timeout=60)
        model_name = req.model or settings.default_importance_model
        return ScoreResponse(score=score, model=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run():
    import uvicorn

    uvicorn.run(
        "sekha_llm_bridge.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
