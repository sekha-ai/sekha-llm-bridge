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
