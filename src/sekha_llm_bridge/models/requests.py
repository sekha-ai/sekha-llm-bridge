"""API request models"""

from pydantic import BaseModel, Field
from typing import List, Optional


class EmbedRequest(BaseModel):
    """Request to generate embedding"""
    text: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Embedding model to use")


class SummarizeRequest(BaseModel):
    """Request to generate summary"""
    messages: List[str] = Field(..., description="Messages to summarize")
    level: str = Field("daily", description="Summary level: daily, weekly, monthly")
    model: Optional[str] = Field(None, description="LLM model to use")


class ExtractRequest(BaseModel):
    """Request to extract entities"""
    text: str = Field(..., description="Text to extract entities from")
    model: Optional[str] = Field(None, description="LLM model to use")


class ScoreRequest(BaseModel):
    """Request to score importance"""
    conversation: str = Field(..., description="Conversation text to score")
    model: Optional[str] = Field(None, description="LLM model to use")


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    model: Optional[str] = Field(None, description="Model to use for completion")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2000, ge=1, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
