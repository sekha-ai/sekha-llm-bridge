# Changelog

All notable changes to sekha-llm-bridge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/sekha-ai/sekha-llm-bridge/releases/tag/v0.1.0