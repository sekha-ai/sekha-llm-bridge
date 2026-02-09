# End-to-End (E2E) Tests

True end-to-end tests that verify the complete sekha-llm-bridge system in a real runtime environment.

## Overview

E2E tests require a **running instance** of the sekha-llm-bridge server. These tests:

- Make real HTTP requests to the API
- Exercise actual provider integrations (Ollama, OpenAI, Anthropic, etc.)
- Verify complete workflows from request to response
- Observe real system behavior under various conditions

**⚠️ Important:** E2E tests will **automatically skip** if no server is detected. This ensures CI/CD pipelines can run unit and integration tests without requiring a running server.

## Test Categories

### 1. Full Flow Tests (`test_full_flow.py`)

Tests complete conversation lifecycle:
- Conversation storage and retrieval
- Embedding generation with multiple dimensions
- Search functionality and ranking
- Cost tracking across operations
- Concurrent request handling

### 2. Resilience Tests (`test_resilience.py`)

Tests system resilience and recovery:
- Provider fallback behavior
- Circuit breaker state transitions
- Graceful degradation under constraints
- Data consistency during failures
- Timeout handling

## Running E2E Tests

### Prerequisites

1. **Running Server**
   ```bash
   # Terminal 1: Start the server
   python -m sekha_llm_bridge.main
   ```

2. **Provider Configuration**
   - At least one provider configured (e.g., Ollama running locally)
   - API keys set in environment for cloud providers (optional)

3. **Environment Variables** (optional)
   ```bash
   export SEKHA_BRIDGE_URL="http://localhost:5001"    # Default
   export SEKHA_CONTROLLER_URL="http://localhost:8080" # Default
   export SEKHA_API_KEY="your_api_key"                # Optional
   export TEST_BASE_URL="http://localhost:8000"       # For conftest health check
   ```

### Run All E2E Tests

```bash
# Terminal 2: Run tests
pytest tests/e2e/ -v -s
```

### Run Specific Test Files

```bash
# Run only full flow tests
pytest tests/e2e/test_full_flow.py -v -s

# Run only resilience tests
pytest tests/e2e/test_resilience.py -v -s
```

### Run Specific Tests

```bash
# Run a single test
pytest tests/e2e/test_full_flow.py::test_full_conversation_flow -v -s

# Run tests matching a pattern
pytest tests/e2e/ -k "fallback" -v -s
```

### Run with Markers

```bash
# Run only E2E tests (skips unit/integration)
pytest -m e2e -v -s

# Run slow tests
pytest -m slow -v -s

# Exclude slow tests
pytest -m "not slow" -v -s
```

## Auto-Skip Behavior

### How It Works

The `conftest.py` fixture automatically checks for a running server:

1. Attempts to connect to `{TEST_BASE_URL}/health`
2. If successful (200 OK), tests proceed
3. If connection fails, **all E2E tests are skipped** with a helpful message

### Skip Message Example

```
SKIPPED [1] tests/e2e/conftest.py:XX: 

⚠️  E2E Test Server Not Available
─────────────────────────────────────────────

E2E tests require a running sekha-llm-bridge server at:
  http://localhost:8000

To run E2E tests:

  1. Start the server in a separate terminal:
     $ python -m sekha_llm_bridge.main

  2. Run E2E tests:
     $ pytest tests/e2e/ -v

  Or run with a custom URL:
     $ TEST_BASE_URL=http://localhost:9000 pytest tests/e2e/ -v

Note: Unit and integration tests do not require a running server.
      Run them with: pytest tests/unit/ tests/integration/ -v
```

## Configuration

### Custom Server URL

If your server runs on a different port:

```bash
TEST_BASE_URL=http://localhost:9000 pytest tests/e2e/ -v
```

### Remote Server

Test against a deployed instance:

```bash
TEST_BASE_URL=https://api.example.com pytest tests/e2e/ -v
```

### Custom Timeouts

The async_client fixture uses a 30-second timeout by default. To customize:

1. Edit `tests/e2e/conftest.py`
2. Modify the `timeout` parameter in the `async_client` fixture

## CI/CD Integration

### GitHub Actions Example

```yaml
# Run unit and integration tests (no server required)
- name: Run Unit and Integration Tests
  run: |
    pytest tests/unit/ tests/integration/ -v --cov

# E2E tests are skipped automatically if server not running
- name: Run All Tests (E2E skipped)
  run: |
    pytest tests/ -v
```

### Running E2E in CI (Optional)

If you want to run E2E tests in CI:

```yaml
- name: Start Server
  run: |
    python -m sekha_llm_bridge.main &
    sleep 5  # Wait for server startup

- name: Run E2E Tests
  run: |
    pytest tests/e2e/ -v -s
  env:
    TEST_BASE_URL: http://localhost:8000

- name: Stop Server
  run: |
    pkill -f "sekha_llm_bridge.main"
```

## Troubleshooting

### Tests Are Skipped

**Symptom:** All E2E tests show as `SKIPPED`

**Solution:**
1. Verify server is running:
   ```bash
   curl http://localhost:8000/health
   ```
2. Check the URL in `TEST_BASE_URL` matches your server
3. Ensure no firewall blocking localhost connections

### Connection Errors

**Symptom:** `httpx.ConnectError: All connection attempts failed`

**Solution:**
1. Confirm server is listening on the correct port:
   ```bash
   netstat -an | grep 8000
   ```
2. Check server logs for startup errors
3. Verify no other service is using the port

### Tests Timeout

**Symptom:** Tests hang or timeout after 30 seconds

**Solution:**
1. Check if providers are responding (especially Ollama)
2. Verify network connectivity to cloud providers
3. Review circuit breaker states: `GET /api/v1/health/providers`

### Provider Not Available

**Symptom:** Test fails with "No suitable providers available"

**Solution:**
1. Ensure at least one provider is configured and healthy
2. For local testing, run Ollama:
   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ```
3. Check provider health:
   ```bash
   curl http://localhost:8000/api/v1/health/providers
   ```

## Best Practices

### Development Workflow

1. **During Development:**
   ```bash
   # Terminal 1: Server with auto-reload
   uvicorn sekha_llm_bridge.main:app --reload
   
   # Terminal 2: Run tests in watch mode
   pytest-watch tests/e2e/ -v -s
   ```

2. **Before Commit:**
   ```bash
   # Run all test types
   pytest tests/ -v --cov
   ```

3. **Pre-PR Checklist:**
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] E2E tests pass (with server running)
   - [ ] Coverage meets threshold

### Writing New E2E Tests

1. **Use the shared fixtures** from `conftest.py`:
   ```python
   @pytest.mark.e2e
   @pytest.mark.asyncio
   async def test_my_feature(async_client, api_headers):
       # Test automatically skips if server unavailable
       response = await async_client.get("/endpoint", headers=api_headers)
       assert response.status_code == 200
   ```

2. **Mark appropriately:**
   ```python
   @pytest.mark.e2e        # For all E2E tests
   @pytest.mark.slow       # For tests > 5 seconds
   @pytest.mark.asyncio    # For async tests
   ```

3. **Handle provider variability:**
   ```python
   # Don't assume specific providers
   if response.status_code == 200:
       # Test passed with available provider
   else:
       # Gracefully handle provider unavailability
       pytest.skip("Provider not available")
   ```

4. **Clean up resources:**
   ```python
   # Use fixtures or context managers
   try:
       # Create test resources
       pass
   finally:
       # Clean up (if applicable)
       pass
   ```

## Related Documentation

- [Unit Tests](../unit/README.md) - Fast, isolated component tests
- [Integration Tests](../integration/README.md) - Multi-component tests with mocking
- [Test Strategy](../../docs/testing-strategy.md) - Overall testing approach

## Questions?

If you encounter issues not covered here, check:
1. Server logs: Look for startup errors or provider connection issues
2. Test output: Review the `-s` flag output for diagnostic information
3. Health endpoint: `curl http://localhost:8000/api/v1/health/providers`
