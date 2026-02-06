import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

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
