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