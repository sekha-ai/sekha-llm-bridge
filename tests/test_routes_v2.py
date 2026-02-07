"""Comprehensive tests for API v2 routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sekha_llm_bridge.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_providers_health(self):
        """Test /api/v1/health/providers endpoint."""
        mock_health_data = {
            "ollama": {"status": "healthy", "latency_ms": 100},
            "openai": {"status": "healthy", "latency_ms": 200},
        }
        
        with patch(
            "sekha_llm_bridge.routes_v2.registry.get_provider_health",
            return_value=mock_health_data,
        ):
            response = client.get("/api/v1/health/providers")
            assert response.status_code == 200
            data = response.json()
            # Response structure includes providers nested under "providers" key
            assert "providers" in data
            assert "ollama" in data["providers"]
            assert "openai" in data["providers"]


class TestModelsEndpoint:
    """Test models listing endpoint."""

    def test_list_all_models(self):
        """Test /api/v1/models endpoint."""
        mock_models = [
            {
                "model_id": "llama3.1:8b",
                "provider_id": "ollama",
                "task": "chat_small",
            },
            {
                "model_id": "gpt-4o",
                "provider_id": "openai",
                "task": "chat_smart",
            },
        ]
        
        with patch(
            "sekha_llm_bridge.routes_v2.registry.list_all_models",
            return_value=mock_models,
        ):
            response = client.get("/api/v1/models")
            assert response.status_code == 200
            models = response.json()
            assert len(models) >= 2
            model_ids = [m["model_id"] for m in models]
            assert "llama3.1:8b" in model_ids


class TestTasksEndpoint:
    """Test tasks listing endpoint."""

    def test_list_tasks(self):
        """Test /api/v1/tasks endpoint."""
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        tasks = response.json()
        assert isinstance(tasks, list)
        # Should include standard tasks
        task_names = [t if isinstance(t, str) else t.get("name") for t in tasks]
        assert any("embed" in str(t).lower() for t in task_names)


class TestRouteRequestEndpoint:
    """Test routing request endpoint."""

    def test_route_request_basic(self):
        """Test basic route request."""
        mock_result = MagicMock()
        mock_result.provider.provider_id = "ollama"
        mock_result.model_id = "llama3.1:8b"
        mock_result.estimated_cost = 0.0
        mock_result.priority = 1

        # Use AsyncMock for async function
        async_mock = AsyncMock(return_value=mock_result)
        
        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            async_mock,
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

    def test_route_request_invalid_task(self):
        """Test route request with invalid task."""
        request_data = {
            "task": "invalid_task_name",
            "require_vision": False,
        }

        response = client.post("/api/v1/route", json=request_data)
        # Invalid enum value returns 500 or 422
        assert response.status_code in [422, 500]

    def test_route_request_with_vision(self):
        """Test route request requiring vision."""
        mock_result = MagicMock()
        mock_result.provider.provider_id = "openai"
        mock_result.model_id = "gpt-4o"
        mock_result.estimated_cost = 0.01
        mock_result.priority = 1

        # Use AsyncMock for async function
        async_mock = AsyncMock(return_value=mock_result)
        
        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            async_mock,
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

    def test_route_request_with_cost_limit(self):
        """Test route request with cost limit."""
        mock_result = MagicMock()
        mock_result.provider.provider_id = "ollama"
        mock_result.model_id = "llama3.1:8b"
        mock_result.estimated_cost = 0.0
        mock_result.priority = 2

        # Use AsyncMock for async function
        async_mock = AsyncMock(return_value=mock_result)
        
        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            async_mock,
        ):
            request_data = {
                "task": "chat_smart",
                "max_cost": 0.001,  # Very tight budget
            }

            response = client.post("/api/v1/route", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["estimated_cost"] <= 0.001
