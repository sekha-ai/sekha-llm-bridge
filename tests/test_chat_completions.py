import pytest
from httpx import AsyncClient, ASGITransport
from sekha_llm_bridge.main import app


@pytest.mark.asyncio
async def test_chat_completions_endpoint_exists():
    """Test that /v1/chat/completions endpoint is registered"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "llama3.1:8b",
            },
        )
    # Should not be 404 (endpoint exists)
    assert resp.status_code != 404


@pytest.mark.asyncio
async def test_chat_completions_requires_messages():
    """Test that chat completions requires messages field"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json={"model": "llama3.1:8b"})
    # Should be 422 (validation error) not 404
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_completions_v1_alias():
    """Test that /api/v1/chat/completions also works"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "llama3.1:8b",
            },
        )
    # Should not be 404 (endpoint exists)
    assert resp.status_code != 404


@pytest.mark.asyncio
async def test_chat_completions_accepts_optional_params():
    """Test that chat completions accepts optional parameters"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "llama3.1:8b",
                "temperature": 0.5,
                "max_tokens": 100,
            },
        )
    # Should not be 404 or 422 (endpoint exists and accepts params)
    assert resp.status_code not in [404, 422]
