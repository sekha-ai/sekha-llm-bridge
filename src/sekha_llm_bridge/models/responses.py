"""API response models"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class EmbedResponse(BaseModel):
    """Response from embedding generation"""
    embedding: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used")
    tokens_used: int = Field(..., description="Number of tokens processed")
    dimension: int = Field(..., description="Embedding vector dimension")


class SummarizeResponse(BaseModel):
    """Response from summarization"""
    summary: str = Field(..., description="Generated summary")
    model: str = Field(..., description="Model used")
    tokens_used: int = Field(..., description="Number of tokens processed")
    level: str = Field(..., description="Summary level: daily, weekly, monthly")
    message_count: int = Field(..., description="Number of messages summarized")


class ExtractResponse(BaseModel):
    """Response from entity extraction"""
    entities: Dict[str, List[str]] = Field(..., description="Extracted entities by type")
    model: str = Field(..., description="Model used")


class ScoreResponse(BaseModel):
    """Response from importance scoring"""
    score: float = Field(..., ge=1.0, le=10.0, description="Importance score")
    reasoning: str = Field(..., description="Reasoning for score")
    model: str = Field(..., description="Model used")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    ollama_status: Dict[str, Any] = Field(..., description="Ollama health info")
    models_loaded: List[str] = Field(..., description="Available models")
    timestamp: str = Field(..., description="ISO timestamp")


class ChatCompletionChoice(BaseModel):
    """Single completion choice"""
    index: int = Field(..., description="Choice index")
    message: Dict[str, str] = Field(..., description="Generated message")
    finish_reason: str = Field(..., description="Reason for completion finish")


class ChatCompletionUsage(BaseModel):
    """Token usage information"""
    prompt_tokens: int = Field(..., description="Tokens in prompt")
    completion_tokens: int = Field(..., description="Tokens in completion")
    total_tokens: int = Field(..., description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(..., description="Completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[ChatCompletionChoice] = Field(..., description="Completion choices")
    usage: ChatCompletionUsage = Field(..., description="Token usage")
