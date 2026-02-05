# Testing Guide - Quick Start

This guide helps you quickly run tests for Sekha LLM Bridge v2.0.

## ⚡ Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -e ".[test]"

# 2. Run unit tests (fast, no services needed)
pytest tests/ -m "not integration and not e2e" -v

# 3. Check coverage
pytest tests/ -m "not integration and not e2e" --cov=sekha_llm_bridge --cov-report=term-missing
```

**That's it!** Unit tests should pass without any external services.

---

## 🛠️ Using the Test Runner Script

We provide a convenient script for running different test suites:

```bash
# Make script executable (first time only)
chmod +x scripts/run-tests.sh

# Run unit tests
./scripts/run-tests.sh unit

# Run with coverage report
./scripts/run-tests.sh coverage

# See all options
./scripts/run-tests.sh help
```

### Available Commands

| Command | Description | Requirements |
|---------|-------------|-------------|
| `unit` | Fast unit tests | None |
| `integration` | Tests with Ollama | Ollama running |
| `e2e` | Full stack tests | All services |
| `coverage` | Tests with coverage | None |
| `failed` | Re-run failed tests | None |
| `watch` | Auto-run on changes | pytest-watch |

---

## 🐞 Debugging Failed Tests

### Show detailed error output
```bash
pytest tests/test_mytest.py -vv
```

### Show local variables on failure
```bash
pytest tests/test_mytest.py --showlocals
```

### Drop into debugger on failure
```bash
pytest tests/test_mytest.py --pdb
```

### Stop on first failure
```bash
pytest tests/ -x
```

### Show print statements
```bash
pytest tests/ -s
```

---

## 🧪 Integration Tests (Optional)

Integration tests require Ollama running locally.

### Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull required models (in another terminal)
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 3. Run integration tests
./scripts/run-tests.sh integration

# Or with pytest directly
pytest tests/integration/ -m integration -v
```

### Skip Integration Tests

By default, unit tests skip integration tests:

```bash
# This automatically excludes integration tests
pytest tests/ -m "not integration and not e2e"
```

---

## 📈 Coverage Reports

Generate HTML coverage report:

```bash
./scripts/run-tests.sh coverage

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## 🔍 Running Specific Tests

### By file
```bash
pytest tests/test_resilience.py -v
```

### By class
```bash
pytest tests/test_resilience.py::TestCircuitBreaker -v
```

### By specific test
```bash
pytest tests/test_resilience.py::TestCircuitBreaker::test_opens_after_failure_threshold -v
```

### Using the script
```bash
./scripts/run-tests.sh specific tests/test_resilience.py
./scripts/run-tests.sh specific "tests/test_resilience.py::TestCircuitBreaker::test_opens_after_failure_threshold"
```

---

## ✅ What Should Pass

### Without External Services
✅ **Should pass:**
- All unit tests (`-m "not integration and not e2e"`)
- Circuit breaker tests
- Configuration validation tests
- Cost estimation tests
- Message conversion tests

### With Ollama Running
✅ **Should also pass:**
- Integration tests (`-m integration`)
- Real provider health checks
- Real embedding generation
- Provider routing tests

### With Full Stack
✅ **Should also pass:**
- E2E tests (`-m e2e`)
- Full request flows
- Context injection tests

---

## 🐛 Common Issues

### "No module named 'sekha_llm_bridge'"
```bash
# Install package in development mode
pip install -e .
```

### "Connection refused" in integration tests
```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/version
```

### "pytest: command not found"
```bash
# Install test dependencies
pip install -e ".[test]"
```

### Tests pass locally but fail in CI
- Check markers: ensure integration tests are marked with `@pytest.mark.integration`
- Check mocks: ensure external services are properly mocked
- Check dependencies: ensure all test deps are in `pyproject.toml`

---

## 📚 More Information

For detailed test documentation, see:
- **[tests/README.md](tests/README.md)** - Comprehensive test documentation
- **[pytest.ini](pytest.ini)** - Test configuration
- **[.github/workflows/tests.yml](.github/workflows/tests.yml)** - CI/CD configuration

---

## ✨ Test Markers

Tests are organized with markers:

```bash
# Run only integration tests
pytest -m integration

# Exclude integration tests
pytest -m "not integration"

# Run E2E tests only
pytest -m e2e

# Exclude both integration and e2e
pytest -m "not integration and not e2e"
```

Available markers:
- `unit` - Unit tests (default)
- `integration` - Requires external services
- `e2e` - End-to-end tests
- `slow` - Tests > 1 second

---

## ✨ Best Practices

1. ✅ **Run unit tests before committing**
   ```bash
   ./scripts/run-tests.sh unit
   ```

2. ✅ **Check coverage for new code**
   ```bash
   ./scripts/run-tests.sh coverage
   ```

3. ✅ **Run integration tests before pushing**
   ```bash
   ./scripts/run-tests.sh integration
   ```

4. ✅ **Use watch mode during development**
   ```bash
   ./scripts/run-tests.sh watch
   ```

5. ✅ **Fix failed tests immediately**
   ```bash
   ./scripts/run-tests.sh failed
   ```

---

## 🚀 CI/CD

Tests run automatically on:
- Every push to `feature/*`, `develop`, or `main`
- Every pull request
- Manual workflow dispatch

CI runs:
1. **Unit tests** - Python 3.11 and 3.12
2. **Integration tests** - With Ollama container
3. **Lint checks** - ruff, black, isort, mypy

Check test status:
- View workflow runs in GitHub Actions tab
- Check PR status checks
- Review coverage reports in Codecov

---

## 🎯 Next Steps

1. Run unit tests to verify setup: `./scripts/run-tests.sh unit`
2. Review test structure: `cat tests/README.md`
3. Start Ollama and run integration tests: `./scripts/run-tests.sh integration`
4. Contribute: Add tests for new features following the patterns in `tests/`

---

**Questions?** See [tests/README.md](tests/README.md) for comprehensive documentation.
