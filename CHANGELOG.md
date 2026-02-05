# Changelog

All notable changes to sekha-llm-bridge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-02-05

### Added - Multi-Provider Registry Architecture
- **Provider Registry System**: Central registry for managing multiple LLM providers
  - Support for Ollama, OpenAI, Anthropic, OpenRouter, and more
  - Priority-based provider selection with automatic fallback
  - Dynamic provider health monitoring
  - Model capability caching for fast routing decisions

- **Intelligent Request Routing**:
  - Task-based routing (embedding, chat_small, chat_smart, vision)
  - Cost-aware routing with budget enforcement
  - Vision capability detection and routing
  - Preferred model support with fallback
  - Multi-dimension embedding support (768, 1536, 3072+)

- **Resilience Patterns**:
  - Circuit breaker per provider (configurable thresholds)
  - Automatic provider failover on errors
  - Health check monitoring
  - Retry logic with exponential backoff
  - Rate limit handling

- **Cost Management**:
  - Real-time cost estimation for all models
  - Per-request cost limits (max_cost parameter)
  - Provider-specific budget tracking
  - Free local model prioritization
  - Cost-aware fallback (prefer cheaper when equivalent)

- **Vision Support**:
  - Multi-modal message support (text + images)
  - Image URL and base64 pass-through
  - Vision model detection and routing
  - LiteLLM vision format conversion

- **Multi-Dimension Embeddings**:
  - Automatic dimension detection from models
  - Per-dimension ChromaDB collection management
  - Cross-collection search with result merging
  - Seamless model switching with dimension migration

- **Configuration**:
  - YAML-based provider configuration
  - Environment variable overrides
  - Hot-reload support for provider changes
  - Validation and schema checking

- **Testing & Quality**:
  - Comprehensive integration test suite (100+ tests)
  - Provider fallback tests
  - Cost limit enforcement tests
  - Circuit breaker behavior tests
  - Multi-dimension embedding tests
  - E2E stack integration tests
  - 95%+ test coverage

### Changed - Breaking Changes
- **Configuration Format**: New YAML structure for providers
  - Old: Single provider in environment variables
  - New: Multi-provider list in `config/llm_providers.yaml`
  - See [MIGRATION.md](./MIGRATION.md) for migration guide

- **API Request Format**:
  - Chat requests now support `require_vision` flag
  - Cost limits via `max_cost` parameter
  - Preferred model selection via `preferred_model` parameter
  - Vision messages support multi-modal content arrays

- **Embedding Behavior**:
  - Embeddings now routed based on dimension
  - Automatic collection selection by dimension
  - Multi-collection search replaces single collection

- **Response Format**:
  - Routing responses include `estimated_cost`
  - Provider metadata includes `provider_type`
  - Circuit breaker state in health checks

### Improved
- **Performance**:
  - Model capability caching reduces routing latency
  - Parallel health checks for providers
  - Optimized fallback chain traversal

- **Reliability**:
  - Circuit breakers prevent cascade failures
  - Provider isolation (one failure doesn't affect others)
  - Graceful degradation on partial provider outages

- **Observability**:
  - Structured logging with provider context
  - Health endpoint includes circuit breaker stats
  - Cost tracking per request
  - Provider performance metrics

### Fixed
- Proper error handling for provider timeouts
- Rate limit detection across provider types
- Authentication error classification
- Dimension mismatch validation

### Migration Guide
For upgrading from v1.x to v2.0, see [MIGRATION.md](./MIGRATION.md)

Key breaking changes:
1. Configuration file format change (YAML)
2. API request parameter additions
3. Embedding collection naming convention
4. Health check response structure

### Dependencies
- Added: `pyyaml` for configuration
- Updated: `litellm` to latest (vision support)
- Updated: `pydantic` to v2.x
- Updated: `chromadb` for multi-collection support

---

## [0.1.0] - 2026-01-16

### Added
- FastAPI service for LLM operations
- Embedding generation via Ollama (nomic-embed-text)
- Summarization (daily/weekly/monthly)
- Entity extraction (people, organizations, concepts)
- Importance scoring (1-10 scale)
- Celery + Redis for async background jobs
- Health monitoring and Prometheus metrics
- Comprehensive test suite (82% coverage)
- Docker support with multi-arch builds

### API Endpoints
- POST /api/v1/embed - Generate text embeddings
- POST /api/v1/summarize - Create conversation summaries
- POST /api/v1/extract - Extract entities from text
- POST /api/v1/score - Score conversation importance
- GET /health - Health check
- GET /metrics - Prometheus metrics

[2.0.0]: https://github.com/sekha-ai/sekha-llm-bridge/compare/v0.1.0...v2.0.0
[0.1.0]: https://github.com/sekha-ai/sekha-llm-bridge/releases/tag/v0.1.0
