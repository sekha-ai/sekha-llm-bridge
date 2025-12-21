"""API response models"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class EmbedResponse(BaseModel):
    """Response from embedding generation"""
    embedding: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used")
    dimension: int = Field(..., description="Embedding dimension")


class SummarizeResponse(BaseModel):
    """Response from summarization"""
    summary: str = Field(..., description="Generated summary")
    level: str = Field(..., description="Summary level")
    model: str = Field(..., description="Model used")
    message_count: int = Field(..., description="Number of messages summarized")


class ExtractResponse(BaseModel):
    """Response from entity extraction"""
    entities: Dict[str, List[str]] = Field(..., description="Extracted entities by category")
    model: str = Field(..., description="Model used")


class ScoreResponse(BaseModel):
    """Response from importance scoring"""
    score: float = Field(..., description="Importance score (1-10)")
    reasoning: str = Field(..., description="Reasoning for score")
    model: str = Field(..., description="Model used")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    ollama_status: Dict[str, Any] = Field(..., description="Ollama health info")
    models_loaded: List[str] = Field(..., description="Available models")
