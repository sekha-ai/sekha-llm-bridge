"""Comprehensive tests for Celery background tasks."""

import json
from unittest.mock import Mock, patch

import pytest

from sekha_llm_bridge.tasks import (
    embed_text_task,
    extract_entities_task,
    score_importance_task,
    summarize_messages_task,
)


class TestEmbedTextTask:
    """Test embed_text_task function."""

    def test_embed_text_basic(self):
        """Test basic text embedding."""
        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4], "index": 0}],
            "usage": {"total_tokens": 5},
        }

        with patch(
            "sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response
        ):
            result = embed_text_task("Hello world")

            assert result == [0.1, 0.2, 0.3, 0.4]
            assert len(result) == 4

    def test_embed_text_with_custom_model(self):
        """Test embedding with custom model."""
        mock_response = {
            "data": [{"embedding": [0.5, 0.6], "index": 0}],
        }

        with patch(
            "sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response
        ) as mock_embed:
            embed_text_task("Test text", model="custom-model")

            call_args = mock_embed.call_args
            assert call_args.kwargs["model"] == "custom-model"

    def test_embed_text_uses_default_model(self):
        """Test embedding uses default model from settings."""
        mock_response = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
        }

        with patch(
            "sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response
        ) as mock_embed:
            # Call without specifying model - should use default
            embed_text_task("Test")

            call_args = mock_embed.call_args
            # Default model is "nomic-embed-text" from actual settings
            assert call_args.kwargs["model"] == "nomic-embed-text"

    def test_embed_text_error_handling(self):
        """Test embedding error handling."""
        with patch(
            "sekha_llm_bridge.tasks.litellm.embedding",
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(Exception, match="API Error"):
                embed_text_task("Test")


class TestSummarizeMessagesTask:
    """Test summarize_messages_task function."""

    def test_summarize_messages_basic(self):
        """Test basic message summarization."""
        messages = ["Message 1", "Message 2", "Message 3"]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Summary of messages"}

        with patch(
            "sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response
        ):
            result = summarize_messages_task(messages, "brief")

            assert result == "Summary of messages"


class TestExtractEntitiesTask:
    """Test extract_entities_task function."""

    def test_extract_entities_basic(self):
        """Test basic entity extraction."""
        entities = [
            {"type": "PERSON", "value": "John Doe", "confidence": 0.95},
            {"type": "ORG", "value": "Acme Corp", "confidence": 0.90},
        ]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": json.dumps(entities)}

        with patch(
            "sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response
        ):
            result = extract_entities_task("John Doe works at Acme Corp")

            assert len(result) == 2
            assert result[0]["type"] == "PERSON"
            assert result[1]["type"] == "ORG"


class TestScoreImportanceTask:
    """Test score_importance_task function."""

    def test_score_importance_basic(self):
        """Test basic importance scoring."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "7"}

        with patch(
            "sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response
        ):
            result = score_importance_task("Important decision made today")

            assert result == 7.0
