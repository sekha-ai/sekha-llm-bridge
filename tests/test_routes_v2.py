"""Comprehensive tests for V2 routing API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock

from sekha_llm_bridge.main import app
from sekha_llm_bridge.registry import RoutingResult

client = TestClient(app)


class TestListModelsEndpoint:
    """Test /api/v1/models endpoint."""

    def test_list_models_success(self):
        """Test successful model listing."""
        mock_models = [
            {
                "model_id": "gpt-4o",
                "provider_id": "openai",
                "task": "chat_smart",
                "context_window": 128000,
                "dimension": None,
                "supports_vision": True,
                "supports_audio": False,
            },
            {
                "model_id": "llama3.1:8b",
                "provider_id": "ollama",
                "task": "chat_small",
                "context_window": 8192,
                "dimension": None,
                "supports_vision": False,
                "supports_audio": False,
            },
        ]

        with patch("sekha_llm_bridge.routes_v2.registry.list_all_models") as mock_list:
            mock_list.return_value = mock_models

            response = client.get("/api/v1/models")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["model_id"] == "gpt-4o"
            assert data[1]["model_id"] == "llama3.1:8b"

    def test_list_models_empty(self):
        """Test listing models when none are available."""
        with patch(
            "sekha_llm_bridge.routes_v2.registry.list_all_models", return_value=[]
        ):
            response = client.get("/api/v1/models")

            assert response.status_code == 200
            assert response.json() == []

    def test_list_models_server_error(self):
        """Test model listing with server error."""
        with patch(
            "sekha_llm_bridge.routes_v2.registry.list_all_models",
            side_effect=Exception("Database error"),
        ):
            response = client.get("/api/v1/models")

            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]

    def test_list_models_response_schema(self):
        """Test model listing response matches schema."""
        mock_models = [
            {
                "model_id": "gpt-4o",
                "provider_id": "openai",
                "task": "chat_smart",
                "context_window": 128000,
                "dimension": None,
                "supports_vision": True,
                "supports_audio": False,
            }
        ]

        with patch(
            "sekha_llm_bridge.routes_v2.registry.list_all_models",
            return_value=mock_models,
        ):
            response = client.get("/api/v1/models")

            assert response.status_code == 200
            data = response.json()[0]

            required_fields = [
                "model_id",
                "provider_id",
                "task",
                "context_window",
                "dimension",
                "supports_vision",
                "supports_audio",
            ]
            for field in required_fields:
                assert field in data


class TestRouteRequestEndpoint:
    """Test /api/v1/route endpoint."""

    @pytest.mark.asyncio
    async def test_route_request_success(self):
        """Test successful routing request."""
        mock_provider = Mock()
        mock_provider.provider_id = "openai"
        mock_provider.provider_type = "litellm"

        mock_result = RoutingResult(
            provider=mock_provider,
            model_id="gpt-4o",
            estimated_cost=0.025,
            reason="Selected by priority",
        )

        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            new=AsyncMock(return_value=mock_result),
        ):
            response = client.post(
                "/api/v1/route", json={"task": "chat_smart", "max_cost": 0.1}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["provider_id"] == "openai"
            assert data["model_id"] == "gpt-4o"
            assert data["estimated_cost"] == 0.025
            assert data["reason"] == "Selected by priority"
            assert data["provider_type"] == "litellm"

    @pytest.mark.asyncio
    async def test_route_request_with_preferred_model(self):
        """Test routing with preferred model."""
        mock_provider = Mock()
        mock_provider.provider_id = "ollama"
        mock_provider.provider_type = "litellm"

        mock_result = RoutingResult(
            provider=mock_provider,
            model_id="llama3.1:8b",
            estimated_cost=0.0,
            reason="Matched preferred model",
        )

        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            new=AsyncMock(return_value=mock_result),
        ):
            response = client.post(
                "/api/v1/route",
                json={"task": "chat_small", "preferred_model": "llama3.1:8b"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["reason"] == "Matched preferred model"

    @pytest.mark.asyncio
    async def test_route_request_with_vision_requirement(self):
        """Test routing with vision requirement."""
        mock_provider = Mock()
        mock_provider.provider_id = "openai"
        mock_provider.provider_type = "litellm"

        mock_result = RoutingResult(
            provider=mock_provider,
            model_id="gpt-4o",
            estimated_cost=0.03,
            reason="Vision support required",
        )

        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            new=AsyncMock(return_value=mock_result),
        ):
            response = client.post(
                "/api/v1/route", json={"task": "chat_smart", "require_vision": True}
            )

            assert response.status_code == 200
            assert response.json()["model_id"] == "gpt-4o"

    def test_route_request_invalid_task(self):
        """Test routing with invalid task type."""
        response = client.post("/api/v1/route", json={"task": "invalid_task"})

        assert response.status_code == 400
        assert "Invalid task" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_route_request_no_provider_available(self):
        """Test routing when no provider is available."""
        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            new=AsyncMock(
                side_effect=RuntimeError("No providers available for task")
            ),
        ):
            response = client.post("/api/v1/route", json={"task": "chat_smart"})

            assert response.status_code == 503
            assert "No providers available" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_route_request_server_error(self):
        """Test routing with unexpected server error."""
        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            new=AsyncMock(side_effect=Exception("Unexpected error")),
        ):
            response = client.post("/api/v1/route", json={"task": "chat_smart"})

            assert response.status_code == 500
            assert "Unexpected error" in response.json()["detail"]

    def test_route_request_missing_required_field(self):
        """Test routing without required task field."""
        response = client.post("/api/v1/route", json={})

        assert response.status_code == 422  # Validation error

    def test_route_request_with_all_parameters(self):
        """Test routing with all optional parameters."""
        mock_provider = Mock()
        mock_provider.provider_id = "openai"
        mock_provider.provider_type = "litellm"

        mock_result = RoutingResult(
            provider=mock_provider,
            model_id="gpt-4o",
            estimated_cost=0.02,
            reason="All constraints met",
        )

        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            new=AsyncMock(return_value=mock_result),
        ):
            response = client.post(
                "/api/v1/route",
                json={
                    "task": "chat_smart",
                    "preferred_model": "gpt-4o",
                    "require_vision": True,
                    "max_cost": 0.05,
                    "context_size": 100000,
                },
            )

            assert response.status_code == 200

    def test_route_request_response_schema(self):
        """Test routing response matches schema."""
        mock_provider = Mock()
        mock_provider.provider_id = "openai"
        mock_provider.provider_type = "litellm"

        mock_result = RoutingResult(
            provider=mock_provider,
            model_id="gpt-4o",
            estimated_cost=0.025,
            reason="Test",
        )

        with patch(
            "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
            new=AsyncMock(return_value=mock_result),
        ):
            response = client.post("/api/v1/route", json={"task": "chat_smart"})

            assert response.status_code == 200
            data = response.json()

            required_fields = [
                "provider_id",
                "model_id",
                "estimated_cost",
                "reason",
                "provider_type",
            ]
            for field in required_fields:
                assert field in data


class TestProviderHealthEndpoint:
    """Test /api/v1/health/providers endpoint."""

    def test_provider_health_success(self):
        """Test successful provider health check."""
        mock_health = {
            "openai": {
                "provider_type": "litellm",
                "circuit_breaker": {"state": "closed", "failure_count": 0},
                "models_count": 5,
            },
            "ollama": {
                "provider_type": "litellm",
                "circuit_breaker": {"state": "closed", "failure_count": 0},
                "models_count": 3,
            },
        }

        with patch(
            "sekha_llm_bridge.routes_v2.registry.get_provider_health",
            return_value=mock_health,
        ):
            response = client.get("/api/v1/health/providers")

            assert response.status_code == 200
            data = response.json()

            assert data["total_providers"] == 2
            assert data["healthy_providers"] == 2
            assert data["total_models"] == 8
            assert "openai" in data["providers"]
            assert "ollama" in data["providers"]

    def test_provider_health_with_unhealthy_provider(self):
        """Test health check with unhealthy provider."""
        mock_health = {
            "openai": {
                "provider_type": "litellm",
                "circuit_breaker": {"state": "closed", "failure_count": 0},
                "models_count": 5,
            },
            "ollama": {
                "provider_type": "litellm",
                "circuit_breaker": {"state": "open", "failure_count": 10},
                "models_count": 3,
            },
        }

        with patch(
            "sekha_llm_bridge.routes_v2.registry.get_provider_health",
            return_value=mock_health,
        ):
            response = client.get("/api/v1/health/providers")

            assert response.status_code == 200
            data = response.json()

            # Only openai is healthy (closed circuit breaker)
            assert data["healthy_providers"] == 1

    def test_provider_health_no_providers(self):
        """Test health check with no providers."""
        with patch(
            "sekha_llm_bridge.routes_v2.registry.get_provider_health", return_value={}
        ):
            response = client.get("/api/v1/health/providers")

            assert response.status_code == 200
            data = response.json()

            assert data["total_providers"] == 0
            assert data["healthy_providers"] == 0
            assert data["total_models"] == 0

    def test_provider_health_server_error(self):
        """Test health check with server error."""
        with patch(
            "sekha_llm_bridge.routes_v2.registry.get_provider_health",
            side_effect=Exception("Health check failed"),
        ):
            response = client.get("/api/v1/health/providers")

            assert response.status_code == 500
            assert "Health check failed" in response.json()["detail"]

    def test_provider_health_response_schema(self):
        """Test health response matches schema."""
        mock_health = {
            "openai": {
                "provider_type": "litellm",
                "circuit_breaker": {"state": "closed"},
                "models_count": 5,
            }
        }

        with patch(
            "sekha_llm_bridge.routes_v2.registry.get_provider_health",
            return_value=mock_health,
        ):
            response = client.get("/api/v1/health/providers")

            assert response.status_code == 200
            data = response.json()

            required_fields = [
                "providers",
                "total_providers",
                "healthy_providers",
                "total_models",
            ]
            for field in required_fields:
                assert field in data

    def test_provider_health_circuit_breaker_states(self):
        """Test health shows different circuit breaker states."""
        mock_health = {
            "provider1": {
                "provider_type": "litellm",
                "circuit_breaker": {"state": "closed"},
                "models_count": 2,
            },
            "provider2": {
                "provider_type": "litellm",
                "circuit_breaker": {"state": "half-open"},
                "models_count": 2,
            },
            "provider3": {
                "provider_type": "litellm",
                "circuit_breaker": {"state": "open"},
                "models_count": 2,
            },
        }

        with patch(
            "sekha_llm_bridge.routes_v2.registry.get_provider_health",
            return_value=mock_health,
        ):
            response = client.get("/api/v1/health/providers")

            assert response.status_code == 200
            data = response.json()

            # Only provider1 has closed circuit breaker
            assert data["healthy_providers"] == 1


class TestListTasksEndpoint:
    """Test /api/v1/tasks endpoint."""

    def test_list_tasks_success(self):
        """Test successful task listing."""
        response = client.get("/api/v1/tasks")

        assert response.status_code == 200
        data = response.json()

        # Should return list of task type strings
        assert isinstance(data, list)
        assert len(data) > 0
        assert all(isinstance(task, str) for task in data)

    def test_list_tasks_includes_known_tasks(self):
        """Test task list includes known task types."""
        response = client.get("/api/v1/tasks")

        assert response.status_code == 200
        data = response.json()

        # Check for known tasks from ModelTask enum
        known_tasks = ["embedding", "chat_small", "chat_smart"]
        for task in known_tasks:
            assert task in data

    def test_list_tasks_response_type(self):
        """Test task list returns strings."""
        response = client.get("/api/v1/tasks")

        assert response.status_code == 200
        data = response.json()

        # All items should be strings
        assert all(isinstance(item, str) for item in data)

    def test_list_tasks_not_empty(self):
        """Test task list is not empty."""
        response = client.get("/api/v1/tasks")

        assert response.status_code == 200
        data = response.json()

        assert len(data) > 0


class TestAPIIntegration:
    """Test API integration scenarios."""

    def test_full_workflow_list_models_then_route(self):
        """Test complete workflow: list models, then route request."""
        # First, list models
        mock_models = [
            {
                "model_id": "gpt-4o",
                "provider_id": "openai",
                "task": "chat_smart",
                "context_window": 128000,
                "dimension": None,
                "supports_vision": True,
                "supports_audio": False,
            }
        ]

        mock_provider = Mock()
        mock_provider.provider_id = "openai"
        mock_provider.provider_type = "litellm"

        mock_result = RoutingResult(
            provider=mock_provider,
            model_id="gpt-4o",
            estimated_cost=0.025,
            reason="Test",
        )

        with patch(
            "sekha_llm_bridge.routes_v2.registry.list_all_models",
            return_value=mock_models,
        ):
            list_response = client.get("/api/v1/models")
            assert list_response.status_code == 200

            with patch(
                "sekha_llm_bridge.routes_v2.registry.route_with_fallback",
                new=AsyncMock(return_value=mock_result),
            ):
                route_response = client.post(
                    "/api/v1/route", json={"task": "chat_smart"}
                )
                assert route_response.status_code == 200

    def test_api_cors_and_headers(self):
        """Test API returns proper headers."""
        response = client.get("/api/v1/tasks")

        assert response.status_code == 200
        # Check content type
        assert "application/json" in response.headers.get("content-type", "")
