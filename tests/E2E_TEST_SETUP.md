# E2E Test Setup - Quick Start

## 🚀 TL;DR

E2E tests require a running server and **auto-skip if unavailable**.

```bash
# Terminal 1: Start server
python -m sekha_llm_bridge.main

# Terminal 2: Run E2E tests
pytest tests/e2e/ -v -s
```

## ✅ What Changed

E2E tests now automatically detect if the server is running:

- **Server Available** → Tests run normally
- **Server Unavailable** → Tests skip with helpful message

### Benefits

1. **CI/CD Friendly**: Unit and integration tests run without needing a server
2. **Clear Feedback**: Helpful skip messages explain how to run E2E tests
3. **No Configuration**: Works out of the box with sensible defaults

## 📝 Quick Commands

### Run Only Unit/Integration Tests (No Server Required)

```bash
pytest tests/unit/ tests/integration/ -v
```

### Run All Tests (E2E Auto-Skipped If No Server)

```bash
pytest tests/ -v
```

### Run E2E Tests (Requires Server)

```bash
# Start server first
python -m sekha_llm_bridge.main

# Then run E2E tests
pytest tests/e2e/ -v -s
```

### Run Specific E2E Test

```bash
pytest tests/e2e/test_full_flow.py::test_full_conversation_flow -v -s
```

### Run with Custom Server URL

```bash
TEST_BASE_URL=http://localhost:9000 pytest tests/e2e/ -v
```

## 📖 Full Documentation

For comprehensive documentation, see:

- **[E2E Test Documentation](./e2e/README.md)** - Complete guide
- **[conftest.py](./e2e/conftest.py)** - Auto-skip implementation
- **[pytest.ini](../pytest.ini)** - Test configuration

## 👁️ What You'll See

### When Server Is Not Running

```
$ pytest tests/e2e/ -v

tests/e2e/test_full_flow.py::test_full_conversation_flow SKIPPED
tests/e2e/test_full_flow.py::test_multi_dimension_workflow SKIPPED
...

========================== SKIPPED DETAILS ===========================

⚠️  E2E Test Server Not Available
─────────────────────────────────────────

E2E tests require a running sekha-llm-bridge server at:
  http://localhost:8000

To run E2E tests:

  1. Start the server in a separate terminal:
     $ python -m sekha_llm_bridge.main

  2. Run E2E tests:
     $ pytest tests/e2e/ -v
```

### When Server Is Running

```
$ pytest tests/e2e/ -v -s

tests/e2e/test_full_flow.py::test_full_conversation_flow PASSED
tests/e2e/test_full_flow.py::test_multi_dimension_workflow PASSED
tests/e2e/test_full_flow.py::test_cost_tracking_workflow PASSED
...

========================== 10 passed in 15.42s ========================
```

## 🔧 Troubleshooting

### Tests Are Skipped

**Problem:** All E2E tests show `SKIPPED`

**Solution:**
1. Start the server: `python -m sekha_llm_bridge.main`
2. Verify it's running: `curl http://localhost:8000/health`
3. Run tests: `pytest tests/e2e/ -v`

### Connection Errors

**Problem:** `httpx.ConnectError` even though server is running

**Solution:**
1. Check server URL matches: `echo $TEST_BASE_URL`
2. Verify server is on correct port: `netstat -an | grep 8000`
3. Check firewall/network settings

### Provider Errors

**Problem:** "No suitable providers available"

**Solution:**
1. Ensure Ollama is running: `ollama serve`
2. Pull required models: `ollama pull nomic-embed-text`
3. Check provider health: `curl http://localhost:8000/api/v1/health/providers`

## 💡 Pro Tips

### Development Workflow

```bash
# Terminal 1: Server with auto-reload
uvicorn sekha_llm_bridge.main:app --reload --port 8000

# Terminal 2: Watch mode testing
pytest-watch tests/e2e/ -v -s
```

### CI/CD Integration

E2E tests automatically skip in CI if no server is configured:

```yaml
# GitHub Actions example
- name: Run Tests
  run: pytest tests/ -v
  # E2E tests will be skipped automatically
```

### Custom Configuration

```bash
# Test against different server
export TEST_BASE_URL="http://localhost:9000"
export SEKHA_BRIDGE_URL="http://localhost:9001"
export SEKHA_CONTROLLER_URL="http://localhost:9002"

pytest tests/e2e/ -v
```

## 📄 Files Modified

- `tests/e2e/conftest.py` - Server detection and auto-skip logic
- `tests/e2e/test_full_flow.py` - Use shared fixtures
- `tests/e2e/test_resilience.py` - Use shared fixtures
- `tests/e2e/README.md` - Comprehensive documentation
- `pytest.ini` - E2E marker already configured

## ❓ Questions?

See the [full E2E documentation](./e2e/README.md) or check:

1. **Server logs** - Look for startup errors
2. **Health endpoint** - `curl http://localhost:8000/health`
3. **Provider status** - `curl http://localhost:8000/api/v1/health/providers`
