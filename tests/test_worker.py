"""Comprehensive tests for Celery worker integration."""

from unittest.mock import patch

from sekha_llm_bridge.tasks import (
    embed_text_task,
    extract_entities_task,
    score_importance_task,
    summarize_messages_task,
)


class TestCeleryTasks:
    """Test Celery task configuration."""

    def test_all_tasks_have_names(self):
        """Test that all tasks have proper names."""
        # Tasks are plain functions, not Celery task objects
        # They get wrapped by Celery when registered
        tasks = [
            embed_text_task,
            summarize_messages_task,
            extract_entities_task,
            score_importance_task,
        ]

        for task in tasks:
            # Tasks are callable functions
            assert callable(task)
            assert task.__name__.endswith("_task")

    def test_embed_text_task_callable(self):
        """Test that embed_text_task is callable."""
        assert callable(embed_text_task)

    def test_summarize_messages_task_callable(self):
        """Test that summarize_messages_task is callable."""
        assert callable(summarize_messages_task)

    def test_extract_entities_task_callable(self):
        """Test that extract_entities_task is callable."""
        assert callable(extract_entities_task)

    def test_score_importance_task_callable(self):
        """Test that score_importance_task is callable."""
        assert callable(score_importance_task)


class TestTaskExecution:
    """Test task execution behavior."""

    def test_embed_text_execution(self):
        """Test embed_text_task can be executed."""
        mock_response = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
        }

        with patch(
            "sekha_llm_bridge.tasks.litellm.embedding", return_value=mock_response
        ):
            result = embed_text_task("Test text")
            assert result == [0.1, 0.2]
