"""Integration test for bridge health endpoint"""

from fastapi.testclient import TestClient
from sekha_llm_bridge.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama_status" in data
