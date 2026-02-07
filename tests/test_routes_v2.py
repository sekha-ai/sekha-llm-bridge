"""Comprehensive tests for API v2 routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sekha_llm_bridge.main import app
from sekha_llm_bridge import routes_v2

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_registry_initialized():
    """Ensure registry appears initialized for all tests."""
    with patch.object(routes_v2.registry, "_initialized", True):
        yield


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_providers_health(self):
        """Test /api/v1/health/providers endpoint."""
        mock_health_data = {
            "ollama": {
                "status": "healthy",
                "latency_ms": 100,
                "circuit_breaker": {"state": "closed"},
                "models_count": 5,
            },
            "openai": {
                "status": "healthy",
                "latency_ms": 200,
                "circuit_breaker": {"state": "closed"},
                "models_count": 3,
            },
        }
        
        with patch.object(
            routes_v2.registry,
            "get_provider_health",
            return_value=mock_health_data,
        ):
            response = client.get("/api/v1/health/providers")
            assert response.status_code == 200
            data = response.json()
            # Response structure includes providers nested under "providers" key
            assert "providers" in data
            assert "ollama" in data["providers"]
            assert "openai" in data["providers"]
            assert data["total_providers"] == 2
            assert data["healthy_providers"] == 2
            assert data["total_models"] == 8


class TestModelsEndpoint:
    """Test models listing endpoint."""

    def test_list_all_models(self):
        """Test /api/v1/models endpoint."""
        mock_models = [
            {
                "model_id": "llama3.1:8b",
                "provider_id": "ollama",
                "task": "chat_small",
                "context_window": 8000,
                "supports_vision": False,
                "supports_audio": False,
            },
            {
                "model_id": "gpt-4o",
                "provider_id": "openai",
                "task": "chat_smart",
                "context_window": 128000,
                "supports_vision": True,
                "supports_audio": False,
            },
        ]
        
        with patch.object(
            routes_v2.registry,
            "list_all_models",
            return_value=mock_models,
        ):
            response = client.get("/api/v1/models")
            assert response.status_code == 200
            models = response.json()
            assert len(models) == 2
            model_ids = [m["model_id"] for m in models]
            assert "llama3.1:8b" in model_ids
            assert "gpt-4o" in model_ids


class TestTasksEndpoint:
    """Test tasks listing endpoint."""

    def test_list_tasks(self):
        """Test /api/v1/tasks endpoint."""
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        tasks = response.json()
        assert isinstance(tasks, list)
        # Should include standard tasks
        assert "embedding" in tasks
        assert "chat_small" in tasks
        assert "chat_smart" in tasks


class TestRouteRequestEndpoint:
    """Test routing request endpoint."""

    def test_route_request_basic(self):
        """Test basic route request."""
        mock_result = MagicMock()
        mock_result.provider = MagicMock()
        mock_result.provider.provider_id = "ollama"
        mock_result.provider.provider_type = "ollama"
        mock_result.model_id = "llama3.1:8b"
        mock_result.estimated_cost = 0.0
        mock_result.priority = 1
        mock_result.reason = "Best available model"

        with patch.object(
            routes_v2.registry,
            "route_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            request_data = {
                "task": "chat_small",
                "require_vision": False,
                "preferred_model": None,
            }

            response = client.post("/api/v1/route", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["provider_id"] == "ollama"
            assert data["model_id"] == "llama3.1:8b"
            assert data["estimated_cost"] == 0.0
            assert data["provider_type"] == "ollama"

    def test_route_request_invalid_task(self):
        """Test route request with invalid task."""
        request_data = {
            "task": "invalid_task_name",
            "require_vision": False,
        }

        response = client.post("/api/v1/route", json=request_data)
        # Invalid task should return error (422 from Pydantic or 400/500 from handler)
        assert response.status_code in [400, 422, 500]
        # Response should contain error details
        assert "detail" in response.json()

    def test_route_request_with_vision(self):
        """Test route request requiring vision."""
        mock_result = MagicMock()
        mock_result.provider = MagicMock()
        mock_result.provider.provider_id = "openai"
        mock_result.provider.provider_type = "openai"
        mock_result.model_id = "gpt-4o"
        mock_result.estimated_cost = 0.01
        mock_result.priority = 1
        mock_result.reason = "Vision support required"

        with patch.object(
            routes_v2.registry,
            "route_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            request_data = {
                "task": "chat_smart",
                "require_vision": True,
            }

            response = client.post("/api/v1/route", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["provider_id"] == "openai"
            assert data["model_id"] == "gpt-4o"
            assert data["estimated_cost"] == 0.01
            assert data["provider_type"] == "openai"

    def test_route_request_with_cost_limit(self):
        """Test route request with cost limit."""
        mock_result = MagicMock()
        mock_result.provider = MagicMock()
        mock_result.provider.provider_id = "ollama"
        mock_result.provider.provider_type = "ollama"
        mock_result.model_id = "llama3.1:8b"
        mock_result.estimated_cost = 0.0
        mock_result.priority = 2
        mock_result.reason = "Cost-effective option"

        with patch.object(
            routes_v2.registry,
            "route_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            request_data = {
                "task": "chat_smart",
                "max_cost": 0.001,  # Very tight budget
            }

            response = client.post("/api/v1/route", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["estimated_cost"] <= 0.001
            assert data["provider_id"] == "ollama"
