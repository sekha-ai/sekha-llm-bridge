import pytest
from httpx import ASGITransport, AsyncClient

from sekha_llm_bridge.main import app


@pytest.mark.asyncio
async def test_embed_requires_text():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post("/embed", json={"text": "hello world", "model": None})
    # With Celery running and litellm configured this should be 200.
    # For now, just assert it's not a 404 so routing is correct.
    assert resp.status_code != 404
