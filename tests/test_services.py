# fmt: skip
"""Unit tests for LLM Bridge services"""

from unittest.mock import AsyncMock, patch

import pytest

from sekha_llm_bridge.models.requests import (
    EmbedRequest,
    ExtractRequest,
    ScoreRequest,
    SummarizeRequest,
)
from sekha_llm_bridge.services.embedding_service import embedding_service
from sekha_llm_bridge.services.entity_extraction_service import (
    entity_extraction_service,
)
from sekha_llm_bridge.services.importance_scorer import importance_scorer_service
from sekha_llm_bridge.services.summarization_service import summarization_service

# ============================================
# Embedding Service Tests
# ============================================


@pytest.mark.asyncio
async def test_embedding_generation():
    """Test embedding generation"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_embedding",
        new=AsyncMock(),
    ) as mock_embed:
        mock_embed.return_value = [0.1] * 768

        request = EmbedRequest(text="Test text")
        response = await embedding_service.generate_embedding(request)

        assert response.dimension == 768
        assert len(response.embedding) == 768
        assert mock_embed.called


@pytest.mark.asyncio
async def test_batch_embedding_generation():
    """Test batch embedding generation"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_embedding",
        new=AsyncMock(),
    ) as mock_embed:
        mock_embed.return_value = [0.1] * 768

        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await embedding_service.batch_generate_embeddings(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 768 for e in embeddings)
        assert mock_embed.call_count == 3


# ============================================
# Summarization Service Tests
# ============================================


@pytest.mark.asyncio
async def test_daily_summarization():
    """Test daily summary generation"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_completion",
        new=AsyncMock(),
    ) as mock_llm:
        mock_llm.return_value = "Test summary of the conversation"

        request = SummarizeRequest(messages=["Message 1", "Message 2"], level="daily")
        response = await summarization_service.generate_summary(request)

        assert response.summary == "Test summary of the conversation"
        assert response.level == "daily"
        assert response.message_count == 2
        assert mock_llm.called


@pytest.mark.asyncio
async def test_weekly_summarization():
    """Test weekly summary generation"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_completion",
        new=AsyncMock(),
    ) as mock_llm:
        mock_llm.return_value = "Weekly themes and patterns"

        request = SummarizeRequest(
            messages=["Day 1 summary", "Day 2 summary"], level="weekly"
        )
        response = await summarization_service.generate_summary(request)

        assert response.level == "weekly"
        assert mock_llm.called


@pytest.mark.asyncio
async def test_monthly_summarization():
    """Test monthly summary generation"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_completion",
        new=AsyncMock(),
    ) as mock_llm:
        mock_llm.return_value = "Monthly overview"

        request = SummarizeRequest(
            messages=["Week 1", "Week 2", "Week 3", "Week 4"], level="monthly"
        )
        response = await summarization_service.generate_summary(request)

        assert response.level == "monthly"


# ============================================
# Entity Extraction Tests
# ============================================


@pytest.mark.asyncio
async def test_entity_extraction():
    """Test entity extraction"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_completion",
        new=AsyncMock(),
    ) as mock_llm:
        mock_llm.return_value = """```
{
  "people": ["Alice", "Bob"],
  "organizations": ["Microsoft"],
  "technical_terms": ["Python", "FastAPI"],
  "concepts": ["API design"]
}
```"""

        request = ExtractRequest(text="Alice and Bob discussed Python at Microsoft")
        response = await entity_extraction_service.extract_entities(request)

        assert "Alice" in response.entities["people"]
        assert "Bob" in response.entities["people"]
        assert "Microsoft" in response.entities["organizations"]
        assert "Python" in response.entities["technical_terms"]


@pytest.mark.asyncio
async def test_entity_extraction_json_parse_error():
    """Test entity extraction handles malformed JSON"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_completion",
        new=AsyncMock(),
    ) as mock_llm:
        mock_llm.return_value = "Not valid JSON"

        request = ExtractRequest(text="Test text")
        response = await entity_extraction_service.extract_entities(request)

        # Should return empty entities on parse failure
        assert response.entities["people"] == []
        assert response.entities["organizations"] == []


# ============================================
# Importance Scorer Tests
# ============================================


@pytest.mark.asyncio
async def test_importance_scoring():
    """Test importance scoring"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_completion",
        new=AsyncMock(),
    ) as mock_llm:
        mock_llm.return_value = (
            "Score: 8\n\nThis conversation contains important decisions."
        )

        request = ScoreRequest(conversation="Important decision-making conversation")
        response = await importance_scorer_service.score_importance(request)

        assert response.score == 8.0
        assert "decisions" in response.reasoning


@pytest.mark.asyncio
async def test_score_extraction_patterns():
    """Test various score extraction patterns"""
    scorer = importance_scorer_service

    assert scorer._extract_score("Score: 7") == 7.0
    assert scorer._extract_score("8/10 for importance") == 8.0
    assert scorer._extract_score("9.5") == 9.5
    assert scorer._extract_score("No score here") == 5.0  # Default

    # Test clamping
    assert scorer._extract_score("Score: 15") == 10.0  # Max
    assert scorer._extract_score("Score: 0") == 1.0  # Min


@pytest.mark.asyncio
async def test_importance_scoring_failure():
    """Test importance scoring handles failures gracefully"""
    with patch(
        "sekha_llm_bridge.utils.llm_client.llm_client.generate_completion",
        new=AsyncMock(),
    ) as mock_llm:
        mock_llm.side_effect = Exception("LLM error")

        request = ScoreRequest(conversation="Test")
        response = await importance_scorer_service.score_importance(request)

        # Should return default score on failure
        assert response.score == 5.0
        assert "Failed" in response.reasoning
