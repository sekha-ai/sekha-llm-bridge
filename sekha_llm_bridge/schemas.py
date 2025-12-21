from typing import List, Optional
from pydantic import BaseModel


# Embed

class EmbedRequest(BaseModel):
    text: str
    model: Optional[str] = None  # default from settings


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str


# Summarize

class SummarizeRequest(BaseModel):
    messages: List[str]
    level: str  # "daily", "weekly", "monthly"
    model: Optional[str] = None


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    level: str


# Extract Entities

class ExtractRequest(BaseModel):
    text: str
    model: Optional[str] = None


class ExtractedEntity(BaseModel):
    type: str
    value: str
    confidence: float


class ExtractResponse(BaseModel):
    entities: List[ExtractedEntity]
    model: str


# Score Importance

class ScoreRequest(BaseModel):
    text: str
    model: Optional[str] = None


class ScoreResponse(BaseModel):
    score: float  # 1.0-10.0
    model: str
