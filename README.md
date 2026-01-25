# Sekha LLM Bridge

> **LLM Operations Layer for Sekha Memory System**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)

---

## What is Sekha LLM Bridge?

The **LLM Bridge** handles all LLM operations for Sekha:

- ✅ Embedding generation (Ollama, OpenAI, Anthropic)
- ✅ Conversation summarization
- ✅ Label suggestions
- ✅ Async operations with retry logic
- ✅ Provider abstraction layer

**Part of the [Sekha Ecosystem](https://github.com/sekha-ai)**

---

## 📚 Documentation

**Complete documentation: [docs.sekha.dev](https://docs.sekha.dev)**

- [Architecture Overview](https://docs.sekha.dev/architecture/overview/)
- [Deployment Guide](https://docs.sekha.dev/deployment/docker-compose/)
- [Configuration](https://docs.sekha.dev/getting-started/configuration/)

---

## 🚀 Quick Start

### With Docker (Recommended)

```bash
# Part of full Sekha deployment
git clone https://github.com/sekha-ai/sekha-docker.git
cd sekha-docker
docker compose up -d
```

### Standalone

```bash
# Clone
git clone https://github.com/sekha-ai/sekha-llm-bridge.git
cd sekha-llm-bridge

# Install
pip install -r requirements.txt

# Run
python -m sekha_llm_bridge
```

---

## 🏗️ Features

### Current (Ollama)

- Embedding generation (nomic-embed-text)
- Summarization (Llama 3.1 8B)
- Batch processing
- Retry with exponential backoff

### Roadmap (Q1 2026)

- [ ] OpenAI integration
- [ ] Anthropic Claude integration
- [ ] Google Gemini support
- [ ] Custom model support

---

## 🧪 Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Type checking
mypy sekha_llm_bridge/

# Linting
ruff check .
black --check .
```

---

## 🔗 Links

- **Main Repo:** [sekha-controller](https://github.com/sekha-ai/sekha-controller)
- **Docs:** [docs.sekha.dev](https://docs.sekha.dev)
- **Website:** [sekha.dev](https://sekha.dev)
- **Discord:** [discord.gg/sekha](https://discord.gg/sekha)

---

## 📄 License

AGPL-3.0 (Free for personal/educational use)  
Commercial license: [hello@sekha.dev](mailto:hello@sekha.dev)

**[License Details](https://docs.sekha.dev/about/license/)**
