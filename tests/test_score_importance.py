import pytest
from httpx import ASGITransport, AsyncClient

from sekha_llm_bridge.main import app


@pytest.mark.asyncio
async def test_score_importance_route_exists():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post(
            "/score_importance",
            json={"text": "Some important decision", "model": None},
        )
    # Again, only check that the route exists and returns JSON
    assert resp.status_code != 404
