"""Pytest configuration and fixtures for E2E tests.

E2E tests require a running instance of the sekha-llm-bridge server.
These tests will be automatically skipped if the server is not detected.
"""

import os

import httpx
import pytest

# Default server URL for E2E tests
DEFAULT_BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session")
def base_url():
    """Get the base URL for E2E tests from environment or use default."""
    return os.getenv("TEST_BASE_URL", DEFAULT_BASE_URL)


@pytest.fixture(scope="session")
def check_server_running(base_url):
    """Check if the sekha-llm-bridge server is running.

    This fixture attempts to connect to the health endpoint.
    If the server is not available, all E2E tests will be skipped.

    To run E2E tests:
    1. Start the server: python -m sekha_llm_bridge.main
    2. Run tests: pytest tests/e2e/ -v

    Or set a custom URL:
    TEST_BASE_URL=http://localhost:9000 pytest tests/e2e/ -v
    """
    try:
        # Try to connect to health endpoint with short timeout
        response = httpx.get(f"{base_url}/health", timeout=2.0)
        if response.status_code == 200:
            return True
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    # Server not available - skip all E2E tests
    pytest.skip(
        f"\n\n"
        f"⚠️  E2E Test Server Not Available\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"E2E tests require a running sekha-llm-bridge server at:\n"
        f"  {base_url}\n"
        f"\n"
        f"To run E2E tests:\n"
        f"\n"
        f"  1. Start the server in a separate terminal:\n"
        f"     $ python -m sekha_llm_bridge.main\n"
        f"\n"
        f"  2. Run E2E tests:\n"
        f"     $ pytest tests/e2e/ -v\n"
        f"\n"
        f"  Or run with a custom URL:\n"
        f"     $ TEST_BASE_URL=http://localhost:9000 pytest tests/e2e/ -v\n"
        f"\n"
        f"Note: Unit and integration tests do not require a running server.\n"
        f"      Run them with: pytest tests/unit/ tests/integration/ -v\n"
        f"\n"
    )


@pytest.fixture(scope="module")
async def async_client(base_url, check_server_running):
    """Create an async HTTP client for E2E tests.

    This fixture depends on check_server_running, so tests will be
    skipped if the server is not available.
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client
