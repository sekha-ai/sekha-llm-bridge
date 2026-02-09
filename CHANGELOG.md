# Changelog

All notable changes to Sekha LLM Bridge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-10

### Added
- **LiteLLM integration**: Universal adapter supporting 100+ LLM providers
- **OpenRouter support**: Access 400+ models through OpenRouter
- **Multi-provider routing**: Intelligent model selection based on task, cost, and availability
- **Provider health monitoring**: Circuit breakers and health checks for all providers
- **Cost estimation**: Real-time cost calculation for all requests
- **Vision model routing**: Automatic detection and routing of vision-capable models
- **New API endpoints**:
  - `POST /api/v1/route`: Get optimal model for a task
  - `GET /api/v1/models`: List all available models across providers
  - `GET /api/v1/health/providers`: Provider health status
- **Ollama support**: Local model hosting via Ollama
- **Fallback handling**: Automatic failover when primary provider unavailable
- **Retry logic**: Exponential backoff with circuit breakers

### Changed
- **BREAKING**: Now uses LiteLLM for all provider interactions (no direct API clients)
- **BREAKING**: Configuration structure updated for multi-provider support
- Provider registration system for dynamic provider management
- Enhanced error handling with detailed failure reasons
- Improved logging with provider and model details

### Fixed
- Better handling of rate limits across providers
- Improved timeout management
- Fixed model availability detection
- Better error messages for provider failures

### Technical
- All tests passing (unit + integration)
- Full CI/CD pipeline validated
- Production-ready Docker images
- Type checking with mypy
- Linting with ruff

### Configuration

Example configuration for v0.2.0:

```yaml
providers:
  ollama:
    enabled: true
    base_url: "http://ollama:11434"
    
  openrouter:
    enabled: true
    api_key: "sk-or-..."
    base_url: "https://openrouter.ai/api/v1"
    
  openai:
    enabled: false
    api_key: "sk-..."
```

## [0.1.0] - 2026-01-15

### Added
- Initial release
- Basic LLM provider support (Ollama, OpenAI)
- OpenAI-compatible chat completions endpoint
- Simple model selection
- Health monitoring
- Docker support

[0.2.0]: https://github.com/sekha-ai/sekha-llm-bridge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sekha-ai/sekha-llm-bridge/releases/tag/v0.1.0
