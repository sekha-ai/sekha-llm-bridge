# Contributing to Sekha LLM Bridge

Thank you for your interest in contributing to Sekha! This document provides guidelines and instructions for contributing to the LLM Bridge component.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/sekha-llm-bridge.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Setup

### Prerequisites
- Python 3.12+
- pip or uv
- Git
- Docker (for Redis and integration tests)
- Ollama or other LLM provider (for testing)

### Install Development Dependencies
```bash
# Using pip
pip install -e ".[dev]"

# Or using uv (faster)
uv pip install -e ".[dev]"
```

### Environment Setup

Create a `.env` file for local development:

```bash
# LLM Provider Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# OpenRouter (optional)
OPENROUTER_API_KEY=your-key-here

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# API Configuration
API_KEY=dev-test-key-min-32-chars-long
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
```

## Testing

### Run the Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sekha_llm_bridge --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v

# Run integration tests only
pytest tests/integration/

# Run unit tests only
pytest tests/unit/
```

Tests should pass and maintain >85% coverage (our current standard).

### Running the Application Locally

```bash
# Start dependencies (Redis)
docker compose up -d redis

# Start the FastAPI server
python -m sekha_llm_bridge.main

# In another terminal, start Celery worker
celery -A sekha_llm_bridge.worker worker --loglevel=info

# Access the API
curl http://localhost:5001/health
```

## Code Style

- **Formatter:** Black (`black src tests`)
- **Import Sorting:** isort (`isort src tests`)
- **Type Checking:** mypy (`mypy src`)
- **Linting:** Ruff (configured in `pyproject.toml`)

Run all checks:

```bash
black src tests
isort src tests
mypy src
ruff check .
pytest --cov=sekha_llm_bridge
```

### Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install

# Manually run on all files
pre-commit run --all-files
```

## Adding New LLM Providers

The LLM Bridge uses LiteLLM for multi-provider support. To add a new provider:

1. Update `src/sekha_llm_bridge/registry.py` with provider configuration
2. Add provider-specific settings to `src/sekha_llm_bridge/config.py`
3. Add health check logic in `src/sekha_llm_bridge/providers/`
4. Add tests in `tests/providers/`
5. Update documentation in `README.md`

Example:

```python
# In registry.py
def register_anthropic(self):
    if self.config.anthropic_enabled:
        self.providers['anthropic'] = ProviderConfig(
            name='anthropic',
            models=['claude-3-opus', 'claude-3-sonnet'],
            # ...
        )
```

## Pull Request Process

1. **Ensure all tests pass:** `pytest --cov=sekha_llm_bridge`
2. **Ensure code style compliance:** `black`, `isort`, `mypy`, `ruff`
3. **Update documentation if needed** (README, docstrings, type hints)
4. **Add test coverage for new functionality** (aim for >85% coverage)
5. **Update CHANGELOG.md** with your changes under `[Unreleased]`
6. **Submit PR with clear description:**
   - What does this change?
   - Why is it needed?
   - How was it tested?
   - Breaking changes? (if any)
7. **Address review feedback promptly**

## Commit Message Guidelines

- Use clear, descriptive commit messages
- Start with a verb in present tense: "Add feature", "Fix bug", "Update docs"
- Reference related issues: "Fixes #123" or "Related to #456"
- Keep commits focused on a single concern

**Examples:**
```
feat: add support for Google Gemini provider
fix: handle rate limit errors with exponential backoff
docs: update OpenRouter configuration examples
test: add integration tests for streaming completions
chore: upgrade LiteLLM to v1.50.0
```

## Reporting Issues

Use GitHub Issues to report bugs or suggest features.

### Bug Reports

Include:

- Python version (`python --version`)
- LLM Bridge version
- LLM provider being used (Ollama, OpenRouter, etc.)
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback (if applicable)
- Relevant logs from Docker/Celery

### Feature Requests

Include:

- Clear description of the proposed feature
- Use case / motivation
- Example API or configuration
- Willingness to contribute implementation

## Architecture Overview

Understanding the LLM Bridge architecture helps with contributions:

```
┌─────────────────────────────────────────┐
│  FastAPI Application (main.py)          │
│  - REST endpoints (/v1/*, /api/v1/*)    │
│  - Health checks, metrics                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Provider Registry (registry.py)         │
│  - Provider registration                 │
│  - Model routing logic                   │
│  - Health monitoring                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  LiteLLM Integration                     │
│  - Universal LLM adapter                 │
│  - 100+ provider support                 │
│  - Automatic retries & fallbacks         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────┬────────────┬─────────────────┐
│ Ollama   │ OpenRouter │ Other Providers │
└──────────┴────────────┴─────────────────┘
```

**Key Files:**
- `src/sekha_llm_bridge/main.py` - FastAPI app, routes
- `src/sekha_llm_bridge/config.py` - Configuration management
- `src/sekha_llm_bridge/registry.py` - Provider registration
- `src/sekha_llm_bridge/providers/` - Provider-specific logic
- `src/sekha_llm_bridge/resilience.py` - Retry logic, circuit breakers
- `src/sekha_llm_bridge/pricing.py` - Cost calculation
- `src/sekha_llm_bridge/tasks.py` - Celery tasks

## Code of Conduct

Please refer to CODE_OF_CONDUCT.md for our community standards.

## Questions?

- **Discord:** [discord.gg/sekha](https://discord.gg/gZb7U9deKH)
- **GitHub Discussions:** [sekha-ai/sekha-llm-bridge/discussions](https://github.com/sekha-ai/sekha-llm-bridge/discussions)
- **Documentation:** [docs.sekha.dev](https://docs.sekha.dev)

---

**Thank you for contributing to Sekha!** 🚀
