(how controller talks to bridge)# Sekha LLM Bridge – Usage Guide

The Sekha LLM Bridge is a standalone service that isolates all LLM-specific logic from the Rust controller. It exposes a simple HTTP API the controller can call for embeddings, summarization, entity extraction, and importance scoring.

## Running the service

1. Install dependencies:

pip install -r requirements.txt

or

poetry install


2. Configure environment:

cp .env.example .env

edit models, Redis URLs, and provider API keys as needed


3. Start Redis and Celery, then the API:

redis-server &

celery -A sekha_llm_bridge.worker.celery_app worker --loglevel=info

uvicorn sekha_llm_bridge.main:app --host 0.0.0.0 --port 5001


## Health check

curl http://localhost:5001/health


The response reports Redis and Ollama status.[web:44]

## Core endpoints

### POST /embed

Request:

{
"text": "hello world",
"model": "nomic-embed-text"
}

Response:

{
"embedding": [0.01, 0.23, ...],
"model": "nomic-embed-text"
}


### POST /summarize

Request:

{
"messages": ["line 1", "line 2"],
"level": "daily",
"model": "ollama/llama3.1"
}

Response:

{
"summary": "Concise daily summary...",
"model": "ollama/llama3.1",
"level": "daily"
}


### POST /extract_entities

Request:

{
"entities": [
{ "type": "PERSON", "value": "Alice", "confidence": 0.9 },
{ "type": "ORG", "value": "Acme", "confidence": 0.8 },
{ "type": "LOCATION", "value": "Paris", "confidence": 0.85 }
],
"model": "ollama/llama3.1"
}


### POST /score_importance

Request:

{ "text": "We decided to migrate infra to Kubernetes." }


Response:

{
"score": 8.5,
"model": "ollama/llama3.1"
}