# End-to-End (E2E) Tests

Comprehensive E2E tests for Sekha v2.0 multi-provider architecture.

## Overview

These tests verify the complete system behavior across multiple components:
- **Controller** (Rust API)
- **Bridge** (Python LLM routing)
- **ChromaDB** (Vector storage)
- **Ollama** (Local LLM runtime)
- **Cloud Providers** (Optional: OpenAI, Anthropic, etc.)

## Test Files

### `test_full_flow.py`
**Complete conversation lifecycle tests**

- ✅ Store conversation via controller
- ✅ Verify embedding in correct dimension collection
- ✅ Search for stored conversation
- ✅ Retrieve full conversation details
- ✅ Verify optimal model selection
- ✅ Test multi-dimension workflows
- ✅ Verify cost tracking
- ✅ Test search ranking quality
- ✅ Test concurrent operations

### `test_resilience.py`
**Failure handling and recovery tests**

- ✅ Test provider fallback mechanisms
- ✅ Verify circuit breaker opening on failures
- ✅ Test circuit breaker recovery
- ✅ Verify graceful degradation
- ✅ Test data consistency during failures
- ✅ Test timeout handling

## Prerequisites

### Required Services

```bash
# 1. Start full Sekha stack
cd sekha-docker
docker-compose -f docker-compose.v2.yml up -d

# Verify all services are running
docker-compose ps
```

Expected services:
- ✅ sekha-controller (port 8080)
- ✅ sekha-bridge (port 5001)
- ✅ chroma (port 8000)
- ✅ ollama (port 11434)
- ✅ postgres (port 5432)
- ✅ redis (port 6379)

### Environment Variables

```bash
# Required
export SEKHA_CONTROLLER_URL="http://localhost:8080"
export SEKHA_BRIDGE_URL="http://localhost:5001"
export SEKHA_API_KEY="test_key_12345678901234567890123456789012"

# Optional (for cloud provider tests)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Python Dependencies

```bash
cd sekha-llm-bridge
pip install -e ".[dev]"

# Or install specific test dependencies
pip install pytest pytest-asyncio httpx
```

## Running Tests

### Run All E2E Tests

```bash
# From sekha-llm-bridge root
pytest tests/e2e/ -v -m e2e -s
```

### Run Specific Test File

```bash
# Full flow tests only
pytest tests/e2e/test_full_flow.py -v -m e2e -s

# Resilience tests only
pytest tests/e2e/test_resilience.py -v -m e2e -s
```

### Run Specific Test

```bash
# Single test function
pytest tests/e2e/test_full_flow.py::test_full_conversation_flow -v -s

# Test with specific markers
pytest tests/e2e/ -v -m "e2e and not slow" -s
```

### Run with Different Verbosity

```bash
# Minimal output
pytest tests/e2e/ -m e2e -q

# Verbose output (recommended)
pytest tests/e2e/ -m e2e -v -s

# Extra verbose (show all details)
pytest tests/e2e/ -m e2e -vv -s
```

## Test Markers

- `@pytest.mark.e2e` - End-to-end test (requires full stack)
- `@pytest.mark.slow` - Test takes >1 second
- `@pytest.mark.asyncio` - Async test function

## Expected Output

### Successful Run

```
tests/e2e/test_full_flow.py::test_full_conversation_flow 
📝 Step 1: Creating conversation...
✅ Created conversation: a1b2c3d4-...

🔍 Step 2: Searching for conversation...
✅ Found conversation in search results

📖 Step 3: Retrieving conversation a1b2c3d4-...
✅ Retrieved conversation successfully

🎯 Step 4: Verifying routing decisions...
✅ 2 healthy providers available

✅ Full conversation flow test PASSED
PASSED

tests/e2e/test_resilience.py::test_provider_fallback
🔄 Testing provider fallback...

🏛️ Step 1: Checking provider health...
✅ 2/2 providers healthy

🧪 Step 2: Testing routing...
✅ Primary provider: ollama_local
   Model: nomic-embed-text
   Cost: $0.0000

💰 Step 3: Testing cost-based fallback...
✅ Fallback provider: ollama_local
   Model: nomic-embed-text
   Cost: $0.0000
✅ Provider fallback test PASSED
PASSED

============================================
2 passed in 12.34s
============================================
```

### Failed Run (Services Not Available)

```
tests/e2e/test_full_flow.py::test_full_conversation_flow
📝 Step 1: Creating conversation...
E   AssertionError: Failed to create conversation: Connection refused
FAILED

============================================
1 failed in 0.56s
============================================
```

## Troubleshooting

### Issue: Connection Refused

**Symptoms:**
```
Failed to create conversation: Connection refused
```

**Solutions:**
1. Verify services are running:
   ```bash
   docker-compose ps
   curl http://localhost:8080/health
   curl http://localhost:5001/health
   ```

2. Check service logs:
   ```bash
   docker-compose logs controller
   docker-compose logs bridge
   ```

3. Restart services:
   ```bash
   docker-compose restart
   ```

### Issue: API Key Authentication Failed

**Symptoms:**
```
401 Unauthorized
```

**Solutions:**
1. Verify API key is set:
   ```bash
   echo $SEKHA_API_KEY
   ```

2. Check controller configuration:
   ```bash
   docker-compose exec controller cat /app/config.yaml
   ```

3. Use correct default key:
   ```bash
   export SEKHA_API_KEY="test_key_12345678901234567890123456789012"
   ```

### Issue: Tests Timeout

**Symptoms:**
```
pytest timeout exceeded
```

**Solutions:**
1. Increase timeout:
   ```bash
   # In test file or pytest.ini
   timeout = 60
   ```

2. Check for slow providers:
   ```bash
   curl http://localhost:5001/api/v1/health/providers
   ```

3. Verify Ollama is responding:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Issue: Conversation Not Found in Search

**Symptoms:**
```
AssertionError: Conversation not found in search results
```

**Solutions:**
1. Wait longer for embedding:
   ```python
   await asyncio.sleep(5)  # Increase from 2
   ```

2. Check ChromaDB:
   ```bash
   curl http://localhost:8000/api/v1/collections
   ```

3. Verify embedding service:
   ```bash
   docker-compose logs bridge | grep "embed"
   ```

### Issue: Circuit Breaker Always Open

**Symptoms:**
```
All providers showing circuit breaker: open
```

**Solutions:**
1. Check provider health:
   ```bash
   curl http://localhost:11434/api/tags
   curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

2. Reset circuit breakers:
   ```bash
   docker-compose restart bridge
   ```

3. Adjust thresholds in config:
   ```yaml
   circuit_breaker:
     failure_threshold: 10  # More tolerant
     reset_timeout: 30      # Faster recovery
   ```

## Performance Benchmarks

### Expected Test Duration

| Test | Duration | Notes |
|------|----------|-------|
| `test_full_conversation_flow` | 3-5s | Includes 2s sleep for embedding |
| `test_multi_dimension_workflow` | 3-4s | Single conversation |
| `test_cost_tracking_workflow` | 1-2s | Metadata only |
| `test_search_ranking_quality` | 5-8s | Creates 3 conversations |
| `test_concurrent_operations` | 2-3s | 5 parallel requests |
| `test_provider_fallback` | 2-3s | Multiple routing checks |
| `test_circuit_breaker_behavior` | 3-5s | Multiple health checks |
| `test_graceful_degradation` | 1-2s | Error path testing |
| `test_data_consistency_during_failures` | 4-6s | Full CRUD cycle |
| `test_timeout_handling` | 1-2s | Fast metadata check |

**Total:** ~30-40 seconds for full E2E suite

### Resource Usage

During test execution:
- **CPU**: 20-40% (Ollama during embedding)
- **Memory**: 2-4 GB (Ollama + services)
- **Disk I/O**: Moderate (ChromaDB writes)
- **Network**: Minimal (local services)

## CI/CD Integration

### GitHub Actions

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Start Sekha Stack
        run: |
          cd sekha-docker
          docker-compose -f docker-compose.v2.yml up -d
          sleep 30  # Wait for services
      
      - name: Run E2E Tests
        env:
          SEKHA_CONTROLLER_URL: http://localhost:8080
          SEKHA_BRIDGE_URL: http://localhost:5001
          SEKHA_API_KEY: test_key_12345678901234567890123456789012
        run: |
          cd sekha-llm-bridge
          pytest tests/e2e/ -v -m e2e
      
      - name: Collect Logs on Failure
        if: failure()
        run: |
          docker-compose -f sekha-docker/docker-compose.v2.yml logs
```

## Contributing

When adding new E2E tests:

1. **Use descriptive names**: `test_feature_scenario`
2. **Add docstrings**: Explain what the test verifies
3. **Use markers**: `@pytest.mark.e2e` and `@pytest.mark.slow` if needed
4. **Include assertions**: Clear failure messages
5. **Clean up**: Delete test data if possible
6. **Document**: Update this README

## Related Documentation

- [Module 4 README](../../docs/MODULE_4_README.md)
- [E2E Testing Guide](../../../sekha-docker/docs/E2E_TESTING.md)
- [Integration Tests](../integration/)
- [Configuration Guide](../../../sekha-docker/docs/configuration-v2.md)

---

**Module 4 - Task 4.5 & 4.6**: E2E Happy Path and Failure & Recovery Tests  
**Status**: ✅ Complete
