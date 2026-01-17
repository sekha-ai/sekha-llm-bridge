
[![codecov](https://codecov.io/gh/sekha-ai/sekha-llm-bridge/branch/main/graph/badge.svg)](https://codecov.io/gh/sekha-ai/sekha-llm-bridge)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)\
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/sekha-llm-bridge.svg)](https://pypi.org/project/sekha-llm-bridge/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sekha-llm-bridge.svg)](https://pypi.org/project/sekha-llm-bridge/)
[![PyPI version](https://badge.fury.io/py/sekha-llm-bridge.svg)](https://badge.fury.io/py/sekha-llm-bridge)
[![CI](https://github.com/sekha-ai/sekha-llm-bridge/workflows/LLM%20Bridge%20CI/badge.svg)](https://github.com/sekha-ai/sekha-llm-bridge/actions/workflows/ci.yml)
[![Downloads](https://pepy.tech/badge/sekha-llm-bridge)](https://pepy.tech/project/sekha-llm-bridge)
[![Docker](https://img.shields.io/docker/v/sekhaa/sekha-llm-bridge?label=docker)](https://ghcr.io/sekha-ai/sekha-llm-bridge)


# Sekha LLM Bridge

Python service for LLM operations (embeddings, summarization, importance scoring, entity extraction).

FastAPI service providing LLM operations for Project Sekha.

## Features

- **Embedding Generation**: Text-to-vector using Nomic Embed
- **Summarization**: Daily/weekly/monthly summaries
- **Entity Extraction**: Extract people, orgs, concepts
- **Importance Scoring**: Rate conversation importance (1-10)
- **Background Jobs**: Celery + Redis for async processing
- **Health Monitoring**: Prometheus metrics + health checks

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start Ollama (separate terminal)
ollama serve

# 4. Pull required models
ollama pull nomic-embed-text:latest
ollama pull llama3.1:8b

# 5. Start Redis (for background jobs)
docker run -d -p 6379:6379 redis:7-alpine

# 6. Start the bridge
python main.py

# 7. Visit documentation
open http://localhost:5001/docs

# Or with Docker
docker build -t sekha-llm-bridge .
docker run -p 5001:5001 --network host sekha-llm-bridge

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down


API Endpoints

Embedding
curl -X POST http://localhost:5001/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

Summarization
curl -X POST http://localhost:5001/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "messages": ["Message 1", "Message 2"],
    "level": "daily"
  }'

Entity Extraction
curl -X POST http://localhost:5001/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "I met John at Microsoft to discuss Python development."}'

Importance Scoring
curl -X POST http://localhost:5001/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{"conversation": "We decided to migrate to Kubernetes..."}'

Integration with Sekha Controller
Update sekha-controller/src/services/llm_bridge_client.rs:

pub const LLM_BRIDGE_URL: &str = "http://localhost:5001";

┌─────────────────┐
│ Sekha Controller│
│   (Rust Core)   │
└────────┬────────┘
         │ HTTP/gRPC
         ▼
┌─────────────────┐      ┌──────────┐
│   LLM Bridge    │◄────►│  Redis   │
│   (FastAPI)     │      │ (Celery) │
└────────┬────────┘      └──────────┘
         │
         ▼
┌─────────────────┐
│     Ollama      │
│ (Local LLMs)    │
└─────────────────┘


Testing

# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Integration tests
pytest tests/test_integration.py


Monitoring
Metrics: http://localhost:5001/metrics (Prometheus)

Health: http://localhost:5001/health

Docs: http://localhost:5001/docs

License
AGPL-v3


## 🚀 **DEPLOYMENT INSTRUCTIONS**
1. Clone and setup
git clone https://github.com/sekha-ai/sekha-llm-bridge.git
cd sekha-llm-bridge

2. Create virtual environment
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows

3. Install dependencies
pip install -r requirements.txt

4. Configure
cp .env.example .env

5. Start services
Terminal 1: Ollama
ollama serve

Terminal 2: Redis
docker run -d -p 6379:6379 redis:7-alpine

Terminal 3: LLM Bridge
python main.py

6. Test
curl http://localhost:5001/health

## Testing

### Run All Tests
Install test dependencies
pip install pytest pytest-asyncio pytest-cov

Run tests
pytest

With coverage report
pytest --cov=services --cov=utils --cov-report=html

View coverage
open htmlcov/index.html


### Test Individual Components

Test services only
pytest tests/test_services.py -v

Test API endpoints
pytest tests/test_api.py -v

Test specific function
pytest tests/test_services.py::test_embedding_generation -v


### Expected Coverage

- Services: >90%
- API endpoints: >85%
- Utilities: >80%


## Testing

### Run Tests

Install test dependencies
pip install -e ".[dev]"

Run all tests
pytest

Run with coverage
pytest --cov=src/sekha_mcp --cov-report=html

View coverage
open htmlcov/index.html


### Test Individual Tools

Test all tools
pytest tests/test_tools.py -v

Test specific tool
pytest tests/test_tools.py::test_memory_store_success -v

Test client
pytest tests/test_client.py -v


### Manual Testing with Real Controller

1. Start Sekha Controller
cd ../sekha-controller
cargo run

2. In another terminal, test MCP server
cd ../sekha-mcp
python main.py

3. Send test via MCP client (e.g., Claude Desktop)
Or use the test script:
python -c "
import asyncio
from src.sekha_mcp.client import sekha_client

async def test():
result = await sekha_client.search_memory('test query', limit=5)
print(result)

asyncio.run(test())
"

text


### Expected Test Coverage

- Tools: >95%
- Client: >85%
- Server: >80%
