"""V2.0 routing API endpoints for multi-provider support."""

import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from .registry import registry
from .config import ModelTask

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Routing V2"])


# ============================================
# Request/Response Models
# ============================================


class ModelInfo(BaseModel):
    """Information about an available model."""
    model_id: str
    provider_id: str
    task: str
    context_window: int
    dimension: Optional[int] = None
    supports_vision: bool = False
    supports_audio: bool = False


class RoutingRequest(BaseModel):
    """Request for model routing."""
    task: str = Field(..., description="Task type: embedding, chat_small, chat_smart, etc.")
    preferred_model: Optional[str] = Field(None, description="Preferred model ID")
    require_vision: bool = Field(False, description="Require vision support")
    max_cost: Optional[float] = Field(None, description="Maximum cost in USD")
    context_size: Optional[int] = Field(None, description="Required context window size")


class RoutingResponse(BaseModel):
    """Response from model routing."""
    provider_id: str
    model_id: str
    estimated_cost: float
    reason: str
    provider_type: str


class ProviderHealth(BaseModel):
    """Health status of a provider."""
    provider_id: str
    provider_type: str
    status: str  # healthy, unhealthy, degraded
    circuit_breaker_state: str
    models_count: int
    last_failure: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall health response."""
    providers: Dict[str, Any]
    total_providers: int
    healthy_providers: int
    total_models: int


# ============================================
# Endpoints
# ============================================


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models across all providers.
    
    Returns:
        List of models with their capabilities and provider information.
    """
    try:
        models = registry.list_all_models()
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route", response_model=RoutingResponse)
async def route_request(request: RoutingRequest):
    """Route a request to the optimal provider and model.
    
    This endpoint determines the best provider and model to use based on:
    - Task type
    - Model preferences
    - Provider availability (circuit breakers)
    - Cost constraints
    - Required capabilities (vision, audio, etc.)
    
    Args:
        request: Routing request with task and constraints
        
    Returns:
        Routing recommendation with provider, model, and cost estimate
    """
    try:
        # Parse task
        try:
            task = ModelTask(request.task)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task '{request.task}'. Valid tasks: {[t.value for t in ModelTask]}"
            )
        
        # Route the request
        result = await registry.route_with_fallback(
            task=task,
            preferred_model=request.preferred_model,
            require_vision=request.require_vision,
            max_cost=request.max_cost,
        )
        
        return RoutingResponse(
            provider_id=result.provider.provider_id,
            model_id=result.model_id,
            estimated_cost=result.estimated_cost,
            reason=result.reason,
            provider_type=result.provider.provider_type,
        )
        
    except RuntimeError as e:
        # No suitable provider found
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/providers", response_model=HealthResponse)
async def provider_health():
    """Get health status of all providers.
    
    Returns:
        Health information including:
        - Circuit breaker states
        - Provider availability
        - Model counts
        - Recent failures
    """
    try:
        health_data = registry.get_provider_health()
        
        # Count healthy providers
        healthy_count = sum(
            1 for p in health_data.values()
            if p.get("circuit_breaker", {}).get("state") == "closed"
        )
        
        # Total models
        total_models = sum(
            p.get("models_count", 0)
            for p in health_data.values()
        )
        
        return HealthResponse(
            providers=health_data,
            total_providers=len(health_data),
            healthy_providers=healthy_count,
            total_models=total_models,
        )
        
    except Exception as e:
        logger.error(f"Provider health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=List[str])
async def list_tasks():
    """List all available task types.
    
    Returns:
        List of task type strings
    """
    return [task.value for task in ModelTask]
