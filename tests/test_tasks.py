"""Comprehensive tests for Celery background tasks."""

import pytest
from unittest.mock import patch, Mock, MagicMock
import json

from sekha_llm_bridge.tasks import (
    embed_text_task,
    summarize_messages_task,
    extract_entities_task,
    score_importance_task,
)


class TestEmbedTextTask:
    """Test embed_text_task function."""

    def test_embed_text_basic(self):
        """Test basic text embedding."""
        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4], "index": 0}],
            "usage": {"total_tokens": 5},
        }

        with patch("sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response):
            result = embed_text_task("Hello world")

            assert result == [0.1, 0.2, 0.3, 0.4]
            assert len(result) == 4

    def test_embed_text_with_custom_model(self):
        """Test embedding with custom model."""
        mock_response = {
            "data": [{"embedding": [0.5, 0.6], "index": 0}],
        }

        with patch("sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response) as mock_embed:
            result = embed_text_task("Test text", model="custom-model")

            assert result == [0.5, 0.6]
            mock_embed.assert_called_once()
            call_args = mock_embed.call_args
            assert call_args.kwargs["model"] == "custom-model"

    def test_embed_text_uses_default_model(self):
        """Test embedding uses default model from settings."""
        mock_response = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
        }

        with patch("sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response) as mock_embed:
            with patch("sekha_llm_bridge.tasks.settings") as mock_settings:
                mock_settings.embedding_model = "default-embedding-model"
                result = embed_text_task("Test")

                call_args = mock_embed.call_args
                assert call_args.kwargs["model"] == "default-embedding-model"

    def test_embed_text_empty_string(self):
        """Test embedding empty string."""
        mock_response = {
            "data": [{"embedding": [], "index": 0}],
        }

        with patch("sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response):
            result = embed_text_task("")

            assert result == []

    def test_embed_text_long_text(self):
        """Test embedding long text."""
        long_text = "word " * 1000
        mock_response = {
            "data": [{"embedding": [0.1] * 1536, "index": 0}],
        }

        with patch("sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response):
            result = embed_text_task(long_text)

            assert len(result) == 1536

    def test_embed_text_special_characters(self):
        """Test embedding text with special characters."""
        special_text = "Hello! @#$% 你好 🚀"
        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        }

        with patch("sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response):
            result = embed_text_task(special_text)

            assert isinstance(result, list)
            assert len(result) == 3

    def test_embed_text_error_handling(self):
        """Test embedding error handling."""
        with patch("sekha_llm_bridge.tasks.litellm.embedding", side_effect=Exception("API Error")):
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

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = summarize_messages_task(messages, "brief")

            assert result == "Summary of messages"

    def test_summarize_messages_with_custom_model(self):
        """Test summarization with custom model."""
        messages = ["Test message"]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Summary"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            result = summarize_messages_task(messages, "detailed", model="custom-model")

            assert result == "Summary"
            call_args = mock_complete.call_args
            assert call_args.kwargs["model"] == "custom-model"

    def test_summarize_messages_different_levels(self):
        """Test summarization with different detail levels."""
        messages = ["Message"]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Summary"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            # Test brief level
            summarize_messages_task(messages, "brief")
            call_args = mock_complete.call_args
            assert "brief" in call_args.kwargs["messages"][0]["content"].lower()

            # Test detailed level
            summarize_messages_task(messages, "detailed")
            call_args = mock_complete.call_args
            assert "detailed" in call_args.kwargs["messages"][0]["content"].lower()

    def test_summarize_messages_joins_correctly(self):
        """Test messages are joined with newlines."""
        messages = ["Line 1", "Line 2", "Line 3"]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Summary"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            summarize_messages_task(messages, "brief")

            call_args = mock_complete.call_args
            user_message = call_args.kwargs["messages"][1]["content"]
            assert user_message == "Line 1\nLine 2\nLine 3"

    def test_summarize_messages_empty_list(self):
        """Test summarization with empty message list."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "No messages"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = summarize_messages_task([], "brief")

            assert result == "No messages"

    def test_summarize_messages_single_message(self):
        """Test summarization with single message."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Single summary"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = summarize_messages_task(["Single message"], "brief")

            assert result == "Single summary"

    def test_summarize_messages_temperature_setting(self):
        """Test summarization uses correct temperature."""
        messages = ["Test"]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Summary"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            summarize_messages_task(messages, "brief")

            call_args = mock_complete.call_args
            assert call_args.kwargs["temperature"] == 0.2

    def test_summarize_messages_max_tokens(self):
        """Test summarization respects max_tokens."""
        messages = ["Test"]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Summary"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            summarize_messages_task(messages, "brief")

            call_args = mock_complete.call_args
            assert call_args.kwargs["max_tokens"] == 256

    def test_summarize_messages_system_prompt(self):
        """Test summarization includes proper system prompt."""
        messages = ["Test"]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Summary"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            summarize_messages_task(messages, "brief")

            call_args = mock_complete.call_args
            system_msg = call_args.kwargs["messages"][0]
            assert system_msg["role"] == "system"
            assert "summarization" in system_msg["content"].lower()


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

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = extract_entities_task("John Doe works at Acme Corp")

            assert len(result) == 2
            assert result[0]["type"] == "PERSON"
            assert result[1]["type"] == "ORG"

    def test_extract_entities_with_custom_model(self):
        """Test entity extraction with custom model."""
        entities = [{"type": "LOCATION", "value": "NYC", "confidence": 0.88}]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": json.dumps(entities)}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            result = extract_entities_task("Visit NYC", model="custom-model")

            assert len(result) == 1
            call_args = mock_complete.call_args
            assert call_args.kwargs["model"] == "custom-model"

    def test_extract_entities_empty_text(self):
        """Test entity extraction from empty text."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "[]"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = extract_entities_task("")

            assert result == []

    def test_extract_entities_no_entities_found(self):
        """Test when no entities are found."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "[]"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = extract_entities_task("Some text without entities")

            assert result == []

    def test_extract_entities_invalid_json(self):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "not valid json"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = extract_entities_task("Test text")

            assert result == []  # Should return empty list on error

    def test_extract_entities_non_list_json(self):
        """Test handling of JSON that's not a list."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": '{"key": "value"}'}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = extract_entities_task("Test text")

            assert result == []  # Should return empty list when not a list

    def test_extract_entities_temperature_zero(self):
        """Test extraction uses temperature 0 for consistency."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "[]"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            extract_entities_task("Test")

            call_args = mock_complete.call_args
            assert call_args.kwargs["temperature"] == 0.0

    def test_extract_entities_max_tokens(self):
        """Test extraction respects max_tokens limit."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "[]"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            extract_entities_task("Test")

            call_args = mock_complete.call_args
            assert call_args.kwargs["max_tokens"] == 256

    def test_extract_entities_system_prompt(self):
        """Test extraction includes proper system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "[]"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            extract_entities_task("Test")

            call_args = mock_complete.call_args
            system_msg = call_args.kwargs["messages"][0]
            assert system_msg["role"] == "system"
            assert "entities" in system_msg["content"].lower()
            assert "JSON" in system_msg["content"]

    def test_extract_entities_multiple_types(self):
        """Test extracting multiple entity types."""
        entities = [
            {"type": "PERSON", "value": "Alice", "confidence": 0.9},
            {"type": "ORG", "value": "Tech Inc", "confidence": 0.85},
            {"type": "LOCATION", "value": "NYC", "confidence": 0.88},
            {"type": "DATE", "value": "2026", "confidence": 0.95},
        ]
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": json.dumps(entities)}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = extract_entities_task("Alice from Tech Inc in NYC during 2026")

            assert len(result) == 4
            types = [e["type"] for e in result]
            assert "PERSON" in types
            assert "ORG" in types
            assert "LOCATION" in types
            assert "DATE" in types


class TestScoreImportanceTask:
    """Test score_importance_task function."""

    def test_score_importance_basic(self):
        """Test basic importance scoring."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "7"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = score_importance_task("Important decision made today")

            assert result == 7.0

    def test_score_importance_with_custom_model(self):
        """Test importance scoring with custom model."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "8"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            result = score_importance_task("Critical update", model="custom-model")

            assert result == 8.0
            call_args = mock_complete.call_args
            assert call_args.kwargs["model"] == "custom-model"

    def test_score_importance_min_value(self):
        """Test score is clamped to minimum of 1.0."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "0"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = score_importance_task("Trivial message")

            assert result == 1.0  # Should be clamped to 1.0

    def test_score_importance_max_value(self):
        """Test score is clamped to maximum of 10.0."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "15"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = score_importance_task("Critical decision")

            assert result == 10.0  # Should be clamped to 10.0

    def test_score_importance_negative_value(self):
        """Test negative scores are clamped to 1.0."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "-5"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = score_importance_task("Test")

            assert result == 1.0

    def test_score_importance_invalid_response(self):
        """Test handling of non-numeric response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "not a number"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = score_importance_task("Test")

            assert result == 5.0  # Should default to 5.0

    def test_score_importance_float_value(self):
        """Test handling of float values."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "7.5"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = score_importance_task("Test")

            assert result == 7.5

    def test_score_importance_whitespace_handling(self):
        """Test score handles whitespace in response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "  8  \n"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = score_importance_task("Test")

            assert result == 8.0

    def test_score_importance_temperature_zero(self):
        """Test scoring uses temperature 0 for consistency."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "5"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            score_importance_task("Test")

            call_args = mock_complete.call_args
            assert call_args.kwargs["temperature"] == 0.0

    def test_score_importance_max_tokens(self):
        """Test scoring uses minimal max_tokens."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "5"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            score_importance_task("Test")

            call_args = mock_complete.call_args
            assert call_args.kwargs["max_tokens"] == 8

    def test_score_importance_system_prompt(self):
        """Test scoring includes proper system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "5"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response) as mock_complete:
            score_importance_task("Test")

            call_args = mock_complete.call_args
            system_msg = call_args.kwargs["messages"][0]
            assert system_msg["role"] == "system"
            assert "importance" in system_msg["content"].lower()
            assert "1 and 10" in system_msg["content"]

    def test_score_importance_empty_text(self):
        """Test scoring empty text."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "1"}

        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_response):
            result = score_importance_task("")

            assert result == 1.0


class TestTaskIntegration:
    """Test task integration and edge cases."""

    def test_all_tasks_are_celery_tasks(self):
        """Test all task functions are decorated as Celery tasks."""
        assert hasattr(embed_text_task, "delay")
        assert hasattr(summarize_messages_task, "delay")
        assert hasattr(extract_entities_task, "delay")
        assert hasattr(score_importance_task, "delay")

    def test_task_names_are_correct(self):
        """Test task names match expected convention."""
        assert embed_text_task.name == "tasks.embed_text"
        assert summarize_messages_task.name == "tasks.summarize_messages"
        assert extract_entities_task.name == "tasks.extract_entities"
        assert score_importance_task.name == "tasks.score_importance"

    def test_tasks_handle_unicode(self):
        """Test tasks handle unicode text correctly."""
        unicode_text = "Hello 世界 🌍"

        # Test embed
        mock_embed_response = {"data": [{"embedding": [0.1], "index": 0}]}
        with patch("sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_embed_response):
            result = embed_text_task(unicode_text)
            assert isinstance(result, list)

        # Test summarize
        mock_complete_response = Mock()
        mock_complete_response.choices = [Mock()]
        mock_complete_response.choices[0].message = {"content": "Summary"}
        with patch("sekha_llm_bridge.tasks.litellm.completion", return_value=mock_complete_response):
            result = summarize_messages_task([unicode_text], "brief")
            assert isinstance(result, str)

    def test_tasks_with_very_long_input(self):
        """Test tasks handle very long input text."""
        long_text = "word " * 10000  # Very long text

        mock_response = {"data": [{"embedding": [0.1], "index": 0}]}
        with patch("sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response):
            result = embed_text_task(long_text)
            assert isinstance(result, list)
