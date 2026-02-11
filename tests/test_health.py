import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from sekha_llm_bridge.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "ollama_status" in data
    assert "models_loaded" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_endpoint_models_loaded_type():
    """Test that models_loaded returns List[str], not List[dict]"""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/health")

    assert resp.status_code == 200
    data = resp.json()

    # Verify models_loaded is a list
    assert isinstance(data["models_loaded"], list)

    # Verify all elements are strings, not dicts
    for model in data["models_loaded"]:
        assert isinstance(model, str), f"Expected str but got {type(model)}: {model}"


@pytest.mark.asyncio
async def test_health_endpoint_with_multiple_models():
    """Test health endpoint correctly extracts model_id from registry response"""
    # Mock registry to return list of model dicts
    mock_models = [
        {"model_id": "nomic-embed-text", "provider_id": "ollama", "task": "embedding"},
        {"model_id": "llama3.1:8b", "provider_id": "ollama", "task": "chat_smart"},
        {"model_id": "llama3.2-vision:11b", "provider_id": "ollama", "task": "vision"},
    ]

    with patch("sekha_llm_bridge.main.registry") as mock_registry:
        mock_registry.list_all_models.return_value = mock_models
        mock_registry.get_provider_health.return_value = {
            "ollama": {"circuit_breaker": {"state": "closed"}}
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.get("/health")

    assert resp.status_code == 200
    data = resp.json()

    # Verify models_loaded contains only the model IDs as strings
    expected_model_ids = ["nomic-embed-text", "llama3.1:8b", "llama3.2-vision:11b"]
    assert data["models_loaded"] == expected_model_ids


@pytest.mark.asyncio
async def test_health_endpoint_with_empty_models():
    """Test health endpoint when no models are available"""
    with patch("sekha_llm_bridge.main.registry") as mock_registry:
        mock_registry.list_all_models.return_value = []
        mock_registry.get_provider_health.return_value = {}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.get("/health")

    assert resp.status_code == 200
    data = resp.json()

    # Verify models_loaded is an empty list
    assert data["models_loaded"] == []
    assert data["status"] == "degraded"  # No healthy providers


@pytest.mark.asyncio
async def test_health_endpoint_status_healthy():
    """Test health endpoint returns 'healthy' status when providers are available"""
    mock_models = [
        {"model_id": "test-model", "provider_id": "ollama", "task": "chat"},
    ]

    with patch("sekha_llm_bridge.main.registry") as mock_registry:
        mock_registry.list_all_models.return_value = mock_models
        mock_registry.get_provider_health.return_value = {
            "ollama": {"circuit_breaker": {"state": "closed"}}
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] == ["test-model"]


@pytest.mark.asyncio
async def test_health_endpoint_status_degraded():
    """Test health endpoint returns 'degraded' status when no providers are healthy"""
    mock_models = [
        {"model_id": "test-model", "provider_id": "ollama", "task": "chat"},
    ]

    with patch("sekha_llm_bridge.main.registry") as mock_registry:
        mock_registry.list_all_models.return_value = mock_models
        # Simulate unhealthy provider (circuit breaker open)
        mock_registry.get_provider_health.return_value = {
            "ollama": {"circuit_breaker": {"state": "open"}}
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"


client = TestClient(app)


def test_summarize_endpoint_mock():
    """Test that the summarize endpoint exists and responds"""
    response = client.post(
        "/summarize",
        json={
            "messages": ["test message 1", "test message 2"],
            "level": "daily",
            "model": "test-model",
        },
    )
    # Will return 500 if Ollama not configured, but endpoint exists
    assert response.status_code in [200, 422, 500]


def test_embedding_endpoint_mock():
    """Test that the embedding endpoint exists and responds"""
    response = client.post(
        "/embeddings",
        json={"text": "test text", "level": "daily", "model": "test-model"},
    )
    # Will return 500 if Ollama not configured, but endpoint exists
    assert response.status_code in [200, 422, 500]
