# Sekha LLM Bridge

> **Universal LLM Adapter - The Bridge Between Memory and Intelligence**

[![CI](https://github.com/sekha-ai/sekha-llm-bridge/workflows/LLM%20Bridge%20CI/badge.svg)](https://github.com/sekha-ai/sekha-llm-bridge/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org)
[![LiteLLM](https://img.shields.io/badge/powered%20by-LiteLLM-purple.svg)](https://litellm.ai)
[![codecov](https://codecov.io/gh/sekha-ai/sekha-llm-bridge/branch/main/graph/badge.svg)](https://codecov.io/gh/sekha-ai/sekha-llm-bridge)
[![Docker Image](https://img.shields.io/badge/ghcr.io-sekha--llm--bridge-blue)](https://github.com/sekha-ai/sekha-docker/pkgs/container/sekha-llm-bridge)

coming soon:

[![PyPI](https://img.shields.io/pypi/v/sekha-llm-bridge.svg)](https://pypi.org/project/sekha-llm-bridge/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sekha-llm-bridge.svg)](https://pypi.org/project/sekha-llm-bridge/)
[![PyPI version](https://badge.fury.io/py/sekha-llm-bridge.svg)](https://badge.fury.io/py/sekha-llm-bridge)

---

## 🎯 What is Sekha LLM Bridge?

**LLM-Bridge is a REQUIRED component** of the Sekha ecosystem. It acts as the universal adapter layer that enables the [Sekha Controller](https://github.com/sekha-ai/sekha-controller) to work with **any LLM provider** - from local Ollama to cloud services like OpenAI, Anthropic, and Google.

### Why is it Required?

The Controller (Rust) focuses on memory orchestration, storage, and retrieval. LLM-Bridge (Python) handles all LLM-specific operations, providing:

- **Provider Abstraction**: Switch between Ollama, GPT-4, Claude, Gemini without changing Controller code
- **Universal Compatibility**: Powered by [LiteLLM](https://litellm.ai) for 100+ LLM providers
- **Async Processing**: Celery-based task queue for expensive LLM operations
- **Retry Logic**: Automatic retries with exponential backoff for reliability
- **Type Safety**: Pydantic models for request/response validation

---

## 🏗️ Architecture Role

```
┌─────────────────────────────────────────┐
│      Sekha Controller (Rust)            │
│  • Memory Orchestration                 │
│  • Context Assembly                     │
│  • Storage (SQLite + Chroma)            │
└──────────────┬──────────────────────────┘
               │ HTTP Calls
               ▼
┌─────────────────────────────────────────┐
│      LLM-Bridge (Python) ← YOU ARE HERE │
│  • Universal LLM Adapter                │
│  • Embedding Generation                 │
│  • Summarization                        │
│  • Entity Extraction                    │
│  • Importance Scoring                   │
└──────────────┬──────────────────────────┘
               │ LiteLLM
               ▼
    ┌──────────┴────────────┐
    │                       │
    ▼                       ▼
┌─────────┐            ┌──────────┐
│ Ollama  │            │ OpenAI   │
│ (Local) │            │ GPT-4    │
└─────────┘            └──────────┘
    ▼                        ▼
┌─────────┐            ┌──────────┐
│Anthropic│            │  Google  │
│ Claude  │            │  Gemini  │
└─────────┘            └──────────┘
```

**Multi-LLM Workflow Example:**
1. Morning: Use Claude for code review → Sekha captures via Bridge
2. Afternoon: Switch to ChatGPT for docs → Bridge forwards to OpenAI
3. Evening: Use Ollama locally for planning → Bridge uses local LLM
4. **All stored in unified sekha.db** regardless of which LLM was used!

---

## ✨ Features

### Core Services

| Endpoint | Purpose | Used By |
|----------|---------|---------|
| `POST /embed` | Generate embeddings for semantic search | Controller (on conversation storage) |
| `POST /summarize` | Hierarchical summarization (daily/weekly/monthly) | Controller orchestrator |
| `POST /extract` | Extract entities from conversations | Controller (future: auto-labeling) |
| `POST /score` | Score conversation importance (1-10) | Controller pruning engine |
| `POST /v1/chat/completions` | OpenAI-compatible chat endpoint | Proxy (optional component) |

### Current Capabilities

- ✅ **Ollama Integration**: Full support for local LLMs
- ✅ **LiteLLM Powered**: Ready for 100+ providers (OpenAI, Anthropic, etc.)
- ✅ **Async Processing**: Celery task queue for background jobs
- ✅ **Retry Logic**: 3 retries with exponential backoff
- ✅ **Health Monitoring**: `/health` endpoint with model availability checks
- ✅ **Prometheus Metrics**: `/metrics` for observability

### Supported LLM Providers (via LiteLLM)

**Currently Tested:**
- Ollama (nomic-embed-text, llama3.1, etc.)

**Ready to Enable:**
- OpenAI (GPT-4, GPT-3.5-turbo, text-embedding-ada-002)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Google (Gemini Pro, Gemini Flash)
- Cohere (Command, Embed)
- Azure OpenAI
- AWS Bedrock
- [100+ more via LiteLLM](https://docs.litellm.ai/docs/providers)

---

## 🚀 Quick Start

### With Docker (Recommended)

LLM-Bridge is included in the full Sekha stack:

```bash
git clone https://github.com/sekha-ai/sekha-docker.git
cd sekha-docker/docker
cp .env.example .env

# Edit .env to configure your LLM provider
nano .env

docker compose -f docker-compose.prod.yml up -d
```

### Standalone Development

```bash
# Clone
git clone https://github.com/sekha-ai/sekha-llm-bridge.git
cd sekha-llm-bridge

# Install dependencies
pip install -r requirements.txt

# Configure (copy and edit)
cp .env.example .env

# Start Redis (required for Celery)
docker run -d -p 6379:6379 redis:7-alpine

# Run
python -m sekha_llm_bridge.main
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Server
HOST=0.0.0.0
PORT=5001

# Ollama (local LLMs)
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text:latest
SUMMARIZATION_MODEL=llama3.1:8b

# Redis (Celery task queue)
REDIS_URL=redis://localhost:6379/0

# Cloud Providers (optional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Logging
LOG_LEVEL=INFO
```

### Using Different LLM Providers

**Switch to OpenAI:**
```bash
EMBEDDING_MODEL=text-embedding-3-small
SUMMARIZATION_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

**Switch to Claude:**
```bash
SUMMARIZATION_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...
```

LiteLLM automatically routes to the correct provider based on model name!

---

## 📡 API Reference

### POST /embed
Generate embedding for text.

**Request:**
```json
{
  "text": "What is the meaning of life?",
  "model": "nomic-embed-text:latest"  // optional
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],  // 768-dim vector
  "model": "nomic-embed-text:latest",
  "dimension": 768,
  "tokens_used": 42
}
```

### POST /summarize
Generate hierarchical summary.

**Request:**
```json
{
  "messages": [
    "User discussed Python best practices",
    "Assistant recommended type hints"
  ],
  "level": "daily",  // daily | weekly | monthly
  "model": "llama3.1:8b",  // optional
  "max_words": 200
}
```

**Response:**
```json
{
  "summary": "Discussed Python type hints and best practices...",
  "level": "daily",
  "model": "llama3.1:8b",
  "message_count": 2,
  "tokens_used": 156
}
```

### POST /v1/chat/completions
OpenAI-compatible chat endpoint.

**Request:**
```json
{
  "model": "llama3.1:8b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

**Response:** Standard OpenAI format

---

## 🔧 Development

### Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=sekha_llm_bridge --cov-report=html

# Type checking
mypy src/

# Linting
ruff check .
black --check .
```

### Project Structure

```
sekha-llm-bridge/
├── src/
│   └── sekha_llm_bridge/
│       ├── main.py              # FastAPI app
│       ├── config.py            # Settings
│       ├── models.py            # Pydantic models
│       ├── tasks.py             # Celery tasks
│       ├── services/
│       │   ├── embedding_service.py
│       │   ├── summarization_service.py
│       │   ├── entity_extraction_service.py
│       │   └── importance_scorer.py
│       └── utils/
│           └── llm_client.py    # LiteLLM wrapper
├── tests/
├── requirements.txt
└── pyproject.toml
```

---

## 🤝 Integration with Controller

The Controller calls LLM-Bridge for:

1. **Embedding Generation**: When storing new conversations
   ```rust
   let embedding = llm_bridge.embed_text(&message_content).await?;
   ```

2. **Summarization**: For hierarchical summaries
   ```rust
   let summary = llm_bridge.summarize(messages, "daily").await?;
   ```

3. **Importance Scoring**: For pruning decisions
   ```rust
   let score = llm_bridge.score_importance(&message).await?;
   ```

All operations are **async** and include automatic retries.

---

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:5001/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-25T20:00:00Z",
  "ollama_status": {
    "status": "healthy",
    "models_available": ["nomic-embed-text:latest", "llama3.1:8b"]
  }
}
```

### Prometheus Metrics
```bash
curl http://localhost:5001/metrics
```

---

## 🗺️ Roadmap

### Q1 2026
- [x] Ollama integration
- [x] LiteLLM foundation
- [ ] OpenAI production testing
- [ ] Anthropic Claude integration
- [ ] Google Gemini support

### Q2 2026
- [ ] Multi-provider load balancing
- [ ] Cost tracking per provider
- [ ] Custom model fine-tuning support
- [ ] Streaming responses

---

## 🔗 Related Projects

- **[sekha-controller](https://github.com/sekha-ai/sekha-controller)** - Memory orchestration (Rust)
- **[sekha-proxy](https://github.com/sekha-ai/sekha-proxy)** - Transparent LLM proxy (optional)
- **[sekha-mcp](https://github.com/sekha-ai/sekha-mcp)** - MCP server for Claude Desktop
- **[sekha-docker](https://github.com/sekha-ai/sekha-docker)** - Full stack deployment

---

## 📚 Documentation

**Full docs:** [docs.sekha.dev](https://docs.sekha.dev)

- [Architecture Overview](https://docs.sekha.dev/architecture/overview/)
- [LLM Provider Setup](https://docs.sekha.dev/configuration/llm-providers/)
- [Deployment Guide](https://docs.sekha.dev/deployment/)

---

## 📄 License

AGPL-3.0 - Free for personal and educational use.

Commercial license available: [hello@sekha.dev](mailto:hello@sekha.dev)

**[View License Details](LICENSE)**

---

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/sekha-ai/sekha-llm-bridge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sekha-ai/sekha-controller/discussions)
- **Email**: [dev@sekha.dev](mailto:dev@sekha.dev)

---

**Built with ❤️ by the Sekha team**
