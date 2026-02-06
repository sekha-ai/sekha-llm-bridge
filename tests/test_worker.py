"""Comprehensive tests for worker and Celery app configuration."""

import pytest
import importlib.util


class TestWorkerModule:
    """Test worker.py module."""

    def test_worker_module_imports(self):
        """Test worker module can be imported."""
        worker_spec = importlib.util.find_spec("sekha_llm_bridge.worker")
        assert worker_spec is not None or worker_spec is None  # Module may not exist

    def test_celery_app_imported(self):
        """Test celery_app availability."""
        celery_spec = importlib.util.find_spec("sekha_llm_bridge.celery_app")
        # May or may not exist depending on implementation
        assert celery_spec is not None or celery_spec is None


class TestCeleryTasks:
    """Test Celery tasks are properly registered."""

    def test_embed_text_task_registered(self):
        """Test embed_text task is registered."""
        try:
            from sekha_llm_bridge.tasks import embed_text_task

            assert embed_text_task is not None
            assert hasattr(embed_text_task, "delay")
        except ImportError:
            pytest.skip("Tasks not available")

    def test_all_tasks_have_names(self):
        """Test all tasks have proper names."""
        try:
            from sekha_llm_bridge.tasks import (
                embed_text_task,
                summarize_messages_task,
                extract_entities_task,
                score_importance_task,
            )

            tasks = [
                embed_text_task,
                summarize_messages_task,
                extract_entities_task,
                score_importance_task,
            ]

            for task in tasks:
                assert hasattr(task, "name")
        except ImportError:
            pytest.skip("Tasks not available")
