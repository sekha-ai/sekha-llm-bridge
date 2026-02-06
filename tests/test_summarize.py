import pytest
from httpx import ASGITransport, AsyncClient

from sekha_llm_bridge.main import app


@pytest.mark.asyncio
async def test_summarize_invalid_level():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post(
            "/summarize",
            json={"messages": ["a"], "level": "invalid", "model": None},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert "Must be daily" in body["detail"]
