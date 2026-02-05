# Sekha LLM Bridge Test Suite

Comprehensive test suite for Sekha v2.0 multi-provider LLM architecture.

## Test Structure

```
tests/
├── README.md                      # This file
├── conftest.py                    # Shared fixtures
├── pytest.ini                     # Pytest configuration
│
├── Unit Tests (no external dependencies)
│   ├── test_config.py            # Configuration validation
│   ├── test_resilience.py        # Circuit breaker logic
│   └── test_services.py          # Service layer logic
│
├── Integration Tests (require running services)
│   └── integration/
│       ├── test_provider_routing.py   # Provider fallback and routing
│       ├── test_cost_limits.py        # Cost-aware routing
│       ├── test_embeddings.py         # Multi-dimension embeddings
│       ├── test_vision.py             # Vision model routing
│       └── test_api_health.py         # Health check endpoints
│
├── E2E Tests (full stack validation)
│   ├── test_e2e_stack.py         # Full request flows
│   ├── test_integration_v2.py    # v2.0 API integration
│   └── test_vision.py            # Vision pass-through
│
└── Legacy Tests (backward compatibility)
    ├── test_embed.py
    ├── test_summarize.py
    ├── test_chat_completions.py
    └── test_health.py
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all unit tests (fast, no external dependencies)
pytest tests/ -m "not integration and not e2e"

# Run with coverage
pytest tests/ --cov=sekha_llm_bridge --cov-report=html
```

### Test Categories

#### 1. Unit Tests (Fast, Isolated)
```bash
# All unit tests
pytest tests/ -m "not integration and not e2e" -v

# Specific test files
pytest tests/test_resilience.py -v
pytest tests/test_config.py -v
pytest tests/test_services.py -v

# Circuit breaker tests only
pytest tests/test_resilience.py::TestCircuitBreaker -v
```

**Requirements:** None (uses mocks)

#### 2. Integration Tests (Require Services)
```bash
# All integration tests
pytest tests/integration/ -m integration -v

# Provider routing tests
pytest tests/integration/test_provider_routing.py -v

# Cost limit tests
pytest tests/integration/test_cost_limits.py -v

# Embedding tests
pytest tests/integration/test_embeddings.py -v

# Vision tests
pytest tests/integration/test_vision.py -v
```

**Requirements:**
- Ollama running at `http://localhost:11434`
- Models pulled: `nomic-embed-text`, `llama3.1:8b`

```bash
# Setup for integration tests
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

#### 3. E2E Tests (Full Stack)
```bash
# All E2E tests
pytest tests/ -m e2e -v

# Full stack tests
pytest tests/test_e2e_stack.py -v

# v2.0 integration tests
pytest tests/test_integration_v2.py -v
```

**Requirements:**
- Sekha Controller running (optional, uses mocks by default)
- Ollama running (optional, uses mocks by default)
- ChromaDB running (optional for full E2E)

### Running Specific Test Suites

```bash
# Quick validation (unit tests only, < 5 seconds)
pytest tests/ -m "not integration and not e2e" --tb=short

# Pre-commit tests (unit + fast integration, < 30 seconds)
pytest tests/ -m "not e2e" -x --ff

# Full test suite (all tests, ~2-5 minutes)
pytest tests/ -v

# Parallel execution (requires pytest-xdist)
pytest tests/ -n auto

# Watch mode for development (requires pytest-watch)
ptw tests/ -- -m "not integration and not e2e"
```

## Test Markers

Tests are marked with pytest markers for selective execution:

| Marker | Description | Requirements |
|--------|-------------|--------------|
| `integration` | Requires running Ollama | Ollama running locally |
| `e2e` | Full stack tests | All services running |
| `slow` | Tests that take > 1 second | None |
| `unit` | Pure unit tests | None (default) |

### Using Markers

```bash
# Only integration tests
pytest -m integration

# Exclude integration tests
pytest -m "not integration"

# Integration but not slow
pytest -m "integration and not slow"

# E2E tests only
pytest -m e2e
```

## Test Configuration

### Environment Variables

```bash
# Override default URLs for integration tests
export OLLAMA_BASE_URL="http://localhost:11434"
export CONTROLLER_URL="http://localhost:8080"
export CONTROLLER_API_KEY="test_key_12345678901234567890123456789012"

# Enable debug logging in tests
export PYTEST_DEBUG=1
export LOG_LEVEL=DEBUG
```

### pytest.ini Settings

```ini
[pytest]
markers =
    integration: Tests requiring external services
    e2e: End-to-end tests with full stack
    slow: Tests that take more than 1 second
    unit: Unit tests (default)

# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output options
addopts = 
    -ra
    --strict-markers
    --tb=short
    --asyncio-mode=auto

# Coverage
[coverage:run]
source = sekha_llm_bridge
omit = 
    tests/*
    */__pycache__/*
    */site-packages/*
```

## Key Test Scenarios

### Circuit Breaker Tests
```bash
# Validate circuit breaker state transitions
pytest tests/test_resilience.py::TestCircuitBreaker::test_opens_after_failure_threshold -v
pytest tests/test_resilience.py::TestCircuitBreaker::test_transitions_to_half_open_after_timeout -v
pytest tests/test_resilience.py::TestCircuitBreaker::test_closes_after_success_in_half_open -v
```

### Provider Fallback Tests
```bash
# Test provider routing and fallback
pytest tests/integration/test_provider_routing.py::TestProviderFallback -v
```

### Multi-Dimension Embeddings
```bash
# Test dimension-aware collection routing
pytest tests/integration/test_embeddings.py::TestMultiDimensionEmbeddings -v
```

### Cost Limits
```bash
# Test cost-aware routing
pytest tests/integration/test_cost_limits.py::TestCostLimits -v
```

### Vision Support
```bash
# Test vision message conversion and routing
pytest tests/test_vision.py::TestVisionMessageConversion -v
pytest tests/integration/test_vision.py::TestVisionRouting -v
```

## Writing New Tests

### Test Template

```python
"""Test description."""

import pytest
from sekha_llm_bridge.config import settings


class TestMyFeature:
    """Test suite for my feature."""

    def test_basic_functionality(self):
        """Test basic behavior."""
        # Arrange
        input_data = "test"
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == "expected"

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async behavior."""
        result = await my_async_function()
        assert result is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_with_external_service(self):
        """Test with real Ollama."""
        # Only runs with -m integration flag
        result = await call_ollama()
        assert result.status == "healthy"
```

### Test Fixtures

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def test_config():
    """Mock configuration for tests."""
    return Settings(providers=[...])

@pytest.fixture
async def mock_provider():
    """Mock LLM provider."""
    provider = MagicMock()
    provider.provider_id = "test"
    return provider

@pytest.fixture
def mock_registry(test_config):
    """Mock model registry."""
    registry = ModelRegistry()
    return registry
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e ".[test]"
    
    - name: Run unit tests
      run: |
        pytest tests/ -m "not integration and not e2e" --cov
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hook

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: Run unit tests
        entry: pytest
        args: [tests/, -m, "not integration and not e2e", --tb=short]
        language: system
        pass_filenames: false
```

Install pre-commit hook:
```bash
pip install pre-commit
pre-commit install
```

## Debugging Tests

### Failed Test Investigation

```bash
# Detailed output for failed test
pytest tests/test_mytest.py::test_function -vv

# Show local variables on failure
pytest tests/test_mytest.py --showlocals

# Drop into debugger on failure
pytest tests/test_mytest.py --pdb

# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s
```

### Test Coverage

```bash
# Generate HTML coverage report
pytest tests/ --cov=sekha_llm_bridge --cov-report=html

# Open coverage report
open htmlcov/index.html

# Show missing lines
pytest tests/ --cov=sekha_llm_bridge --cov-report=term-missing

# Fail if coverage below threshold
pytest tests/ --cov=sekha_llm_bridge --cov-fail-under=80
```

## Performance Testing

```bash
# Show test durations
pytest tests/ --durations=10

# Profile slow tests
pytest tests/ --profile

# Show slowest tests
pytest tests/ --durations=0 | head -n 20
```

## Common Issues

### Issue: "No module named 'sekha_llm_bridge'"
**Solution:** Install package in development mode
```bash
pip install -e .
```

### Issue: Integration tests fail with connection errors
**Solution:** Ensure Ollama is running
```bash
ollama serve
# In another terminal:
pytest tests/integration/ -m integration
```

### Issue: "asyncio fixture not found"
**Solution:** Install pytest-asyncio
```bash
pip install pytest-asyncio
```

### Issue: Tests pass locally but fail in CI
**Solution:** Check for environment-specific dependencies
- Ensure all test dependencies in `pyproject.toml`
- Mock external services properly
- Use markers to skip integration tests in CI

## Test Metrics

Current test coverage (as of v2.0):

| Component | Coverage | Tests |
|-----------|----------|-------|
| Circuit Breakers | 98% | 15 |
| Provider Routing | 95% | 25 |
| Cost Estimation | 100% | 12 |
| Vision Support | 92% | 18 |
| Config Validation | 95% | 10 |
| **Overall** | **94%** | **80+** |

## Contributing

When adding new features:

1. ✅ Write unit tests first (TDD)
2. ✅ Add integration tests if using external services
3. ✅ Update this README if adding new test categories
4. ✅ Ensure all tests pass: `pytest tests/ -v`
5. ✅ Check coverage: `pytest tests/ --cov`
6. ✅ Use appropriate markers (`@pytest.mark.integration`, etc.)

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Sekha v2.0 Architecture](../../docs/architecture-v2.md)
- [Contributing Guide](../../CONTRIBUTING.md)

## Questions?

See the main [Sekha documentation](../../README.md) or open an issue on GitHub.
