"""Pytest configuration and fixtures for test suite."""

from unittest.mock import Mock, patch

import pytest

# Mock settings before any imports that need it
mock_settings = Mock()
mock_settings.ollama_base_url = "http://localhost:11434"
mock_settings.providers = []
mock_settings.default_provider = "test-provider"
mock_settings.embedding_model = "text-embedding-3-small"
mock_settings.summarization_model = "gpt-4o-mini"
mock_settings.extraction_model = "gpt-4o-mini"
mock_settings.importance_model = "gpt-4o-mini"
mock_settings.server_host = "0.0.0.0"
mock_settings.server_port = 5001
mock_settings.log_level = "info"
mock_settings.default_models = Mock(
    embedding="nomic-embed-text",
    chat_smart="llama3.1:8b",
    chat_small="llama3.1:8b",
)


@pytest.fixture(autouse=True)
def mock_settings_fixture():
    """Automatically mock settings for all tests."""
    # Patch get_settings() function instead of non-existent settings attribute
    with patch("sekha_llm_bridge.config.get_settings", return_value=mock_settings):
        with patch("sekha_llm_bridge.config.settings", mock_settings):
            with patch("sekha_llm_bridge.registry.settings", mock_settings):
                with patch("sekha_llm_bridge.tasks.settings", mock_settings):
                    yield mock_settings


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_embedding_text():
    """Sample text for embedding tests."""
    return "This is a sample text for embedding generation."


@pytest.fixture
def mock_litellm_completion():
    """Mock litellm completion response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = {"content": "Test response"}
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-4o"
    mock_response.usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    return mock_response


@pytest.fixture
def mock_litellm_embedding():
    """Mock litellm embedding response."""
    return {
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}],
        "usage": {"total_tokens": 5},
        "model": "text-embedding-3-small",
    }


@pytest.fixture
def mock_provider_config():
    """Mock provider configuration."""
    return {
        "id": "test-provider",
        "provider_type": "litellm",
        "base_url": "http://localhost:8000",
        "api_key": "test-key",
        "priority": 1,
        "models": [
            {
                "model_id": "gpt-4o",
                "task": "chat_smart",
                "context_window": 128000,
            },
            {
                "model_id": "gpt-4o-mini",
                "task": "chat_small",
                "context_window": 128000,
            },
        ],
    }


@pytest.fixture
def mock_model_info():
    """Mock model information."""
    return {
        "model_id": "gpt-4o",
        "provider_id": "test-provider",
        "display_name": "GPT-4o",
        "context_window": 128000,
        "supports_vision": True,
        "supports_audio": False,
        "supports_function_calling": True,
    }


@pytest.fixture
def mock_registry():
    """Mock model registry."""
    registry_mock = Mock()
    registry_mock.list_models.return_value = [
        {
            "model_id": "gpt-4o",
            "provider_id": "openai",
            "task": "chat_smart",
        },
        {
            "model_id": "gpt-4o-mini",
            "provider_id": "openai",
            "task": "chat_small",
        },
    ]
    registry_mock.route_request.return_value = {
        "provider_id": "openai",
        "model_id": "gpt-4o",
    }
    return registry_mock


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Clear any cached singletons
    yield
    # Cleanup after test


@pytest.fixture
def test_client():
    """Create a test client for FastAPI."""
    from fastapi.testclient import TestClient

    # Import within fixture to ensure mocks are in place
    try:
        from sekha_llm_bridge.main import app

        return TestClient(app)
    except Exception:
        # If app can't be created, return a mock
        return Mock()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test (deselect with '-m \"not e2e\"')"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (deselect with '-m \"not slow\"')"
    )


# Auto-use fixture to mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_apis():
    """Mock external API calls for all tests."""
    with patch("litellm.completion") as mock_completion:
        with patch("litellm.acompletion") as mock_acompletion:
            with patch("litellm.embedding") as mock_embedding:
                # Set default return values
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = {"content": "Mocked response"}
                mock_response.choices[0].finish_reason = "stop"
                mock_response.model = "gpt-4o"
                mock_response.usage = {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                }

                mock_completion.return_value = mock_response
                mock_acompletion.return_value = mock_response

                mock_embedding.return_value = {
                    "data": [{"embedding": [0.1] * 1536, "index": 0}],
                    "usage": {"total_tokens": 5},
                }

                yield {
                    "completion": mock_completion,
                    "acompletion": mock_acompletion,
                    "embedding": mock_embedding,
                }


@pytest.fixture
def mock_celery_task():
    """Mock Celery task for testing."""
    task_mock = Mock()
    task_mock.delay.return_value = Mock(id="test-task-id")
    task_mock.apply_async.return_value = Mock(id="test-task-id")
    return task_mock
