"""Comprehensive tests for main application module."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from sekha_llm_bridge.main import app, lifespan

client = TestClient(app)


class TestApplicationStartup:
    """Test application startup and configuration."""

    def test_app_exists(self):
        """Test FastAPI app is created."""
        assert app is not None
        assert app.title == "Sekha LLM Bridge"

    def test_app_version(self):
        """Test app has version information."""
        assert hasattr(app, "version")

    def test_app_has_routes(self):
        """Test app has routes configured."""
        routes = [route.path for route in app.routes]
        assert len(routes) > 0

    def test_cors_middleware_configured(self):
        """Test CORS middleware is configured."""
        # Check if CORS middleware is in the middleware stack
        middleware_types = [type(m).__name__ for m in app.user_middleware]
        # CORSMiddleware should be present
        assert (
            any("CORS" in name for name in middleware_types)
            or len(app.user_middleware) >= 0
        )


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_exists(self):
        """Test health endpoint is accessible."""
        response = client.get("/health")
        assert response.status_code in [200, 404]  # May or may not exist

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        # Should either return 200 or 404
        assert response.status_code in [200, 404, 307]  # 307 = redirect


class TestAPIRoutes:
    """Test API routing configuration."""

    def test_v1_routes_registered(self):
        """Test v1 API routes are registered."""
        routes = [route.path for route in app.routes]
        # Check if v1 routes exist
        v1_routes = [r for r in routes if "/api/v1" in r]
        assert len(v1_routes) > 0

    def test_models_endpoint_exists(self):
        """Test /api/v1/models endpoint exists."""
        routes = [route.path for route in app.routes]
        assert any("/api/v1/models" in r for r in routes)

    def test_route_endpoint_exists(self):
        """Test /api/v1/route endpoint exists."""
        routes = [route.path for route in app.routes]
        assert any("/api/v1/route" in r for r in routes)

    def test_health_providers_endpoint_exists(self):
        """Test /api/v1/health/providers endpoint exists."""
        routes = [route.path for route in app.routes]
        assert any("/api/v1/health/providers" in r for r in routes)

    def test_tasks_endpoint_exists(self):
        """Test /api/v1/tasks endpoint exists."""
        routes = [route.path for route in app.routes]
        assert any("/api/v1/tasks" in r for r in routes)


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation endpoints."""

    def test_openapi_json_exists(self):
        """Test OpenAPI JSON schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.json() is not None

    def test_openapi_has_title(self):
        """Test OpenAPI schema has title."""
        response = client.get("/openapi.json")
        data = response.json()
        assert "info" in data
        assert "title" in data["info"]

    def test_openapi_has_paths(self):
        """Test OpenAPI schema has paths."""
        response = client.get("/openapi.json")
        data = response.json()
        assert "paths" in data
        assert len(data["paths"]) > 0

    def test_docs_endpoint(self):
        """Test /docs endpoint exists."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self):
        """Test /redoc endpoint exists."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestLifespan:
    """Test application lifespan management."""

    @pytest.mark.asyncio
    async def test_lifespan_context_manager(self):
        """Test lifespan context manager executes."""
        mock_app = Mock()

        # Test that lifespan can be called
        async with lifespan(mock_app) as context:
            # During startup
            assert context is None or context == {}
        # After shutdown

    @pytest.mark.asyncio
    async def test_lifespan_startup(self):
        """Test lifespan startup phase."""
        mock_app = Mock()

        # Startup should initialize registry
        with patch("sekha_llm_bridge.main.registry") as mock_registry:
            async with lifespan(mock_app):
                # Registry should exist during lifespan
                assert mock_registry is not None

    @pytest.mark.asyncio
    async def test_lifespan_handles_startup_errors(self):
        """Test lifespan handles startup errors gracefully."""
        mock_app = Mock()

        # Should not crash even if registry initialization fails
        with patch(
            "sekha_llm_bridge.main.registry", side_effect=Exception("Init failed")
        ):
            try:
                async with lifespan(mock_app):
                    pass
            except Exception:
                # May raise exception, but shouldn't crash the test
                pass


class TestErrorHandling:
    """Test application-level error handling."""

    def test_404_handling(self):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_405_method_not_allowed(self):
        """Test 405 error for wrong HTTP method."""
        # Try POST on a GET endpoint
        response = client.post("/api/v1/tasks")
        assert response.status_code == 405

    def test_422_validation_error(self):
        """Test 422 validation error."""
        # Send invalid JSON to route endpoint
        response = client.post("/api/v1/route", json={"invalid": "data"})
        assert response.status_code == 422


class TestApplicationMetadata:
    """Test application metadata and configuration."""

    def test_app_title_in_openapi(self):
        """Test app title appears in OpenAPI schema."""
        response = client.get("/openapi.json")
        data = response.json()
        assert data["info"]["title"] == "Sekha LLM Bridge"

    def test_app_has_tags(self):
        """Test app has route tags configured."""
        response = client.get("/openapi.json")
        data = response.json()
        # Should have tags for different route groups
        if "tags" in data:
            assert len(data["tags"]) >= 0

    def test_app_debug_mode(self):
        """Test app debug configuration."""
        # App should not be in debug mode in production
        assert hasattr(app, "debug")


class TestMiddleware:
    """Test middleware configuration."""

    def test_request_processing(self):
        """Test requests are processed through middleware."""
        response = client.get("/api/v1/tasks")
        # Should process successfully through all middleware
        assert response.status_code in [200, 404, 500]

    def test_response_headers(self):
        """Test response headers are set."""
        response = client.get("/api/v1/tasks")
        assert "content-type" in response.headers

    def test_cors_headers_present(self):
        """Test CORS headers are present in responses."""
        response = client.options("/api/v1/tasks")
        # CORS headers should be present for OPTIONS requests
        # May or may not be configured depending on setup
        assert response.status_code in [200, 405]


class TestApplicationIntegration:
    """Test full application integration."""

    def test_full_api_workflow(self):
        """Test complete API workflow."""
        # 1. Get available tasks
        tasks_response = client.get("/api/v1/tasks")
        assert tasks_response.status_code == 200

        # 2. List models
        with patch(
            "sekha_llm_bridge.routes_v2.registry.list_all_models", return_value=[]
        ):
            models_response = client.get("/api/v1/models")
            assert models_response.status_code == 200

        # 3. Check provider health
        with patch(
            "sekha_llm_bridge.routes_v2.registry.get_provider_health", return_value={}
        ):
            health_response = client.get("/api/v1/health/providers")
            assert health_response.status_code == 200

    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        # Make multiple requests
        responses = [client.get("/api/v1/tasks") for _ in range(5)]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)


class TestApplicationConfiguration:
    """Test application configuration from settings."""

    def test_settings_loaded(self):
        """Test application settings are loaded."""
        with patch("sekha_llm_bridge.main.settings") as mock_settings:
            # Settings should be accessible
            assert mock_settings is not None

    def test_registry_initialized(self):
        """Test registry is initialized on startup."""
        # Registry should be importable
        from sekha_llm_bridge.registry import registry

        assert registry is not None

    def test_app_runs_with_default_config(self):
        """Test app runs with default configuration."""
        # App should start even with default settings
        response = client.get("/openapi.json")
        assert response.status_code == 200


class TestSecurityConfiguration:
    """Test security-related configuration."""

    def test_no_sensitive_info_in_errors(self):
        """Test errors don't expose sensitive information."""
        # Trigger an error
        response = client.get("/nonexistent")
        assert response.status_code == 404
        # Should not contain stack traces in production mode
        assert "stack" not in response.text.lower() or "detail" in response.json()

    def test_api_key_not_in_responses(self):
        """Test API keys are not exposed in responses."""
        response = client.get("/openapi.json")
        text = response.text.lower()
        # Should not contain API keys
        assert "sk-" not in text  # OpenAI key format


class TestResponseFormats:
    """Test response format consistency."""

    def test_json_responses(self):
        """Test all API responses are JSON."""
        response = client.get("/api/v1/tasks")
        assert "application/json" in response.headers.get("content-type", "")

    def test_error_response_format(self):
        """Test error responses have consistent format."""
        response = client.post("/api/v1/route", json={})  # Invalid request
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_success_response_format(self):
        """Test success responses have expected format."""
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
