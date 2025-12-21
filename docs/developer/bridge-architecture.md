# LLM Bridge Architecture

The LLM Bridge isolates provider-specific logic from the Rust controller and exposes a small HTTP API on port 5001.

## Stack

- **FastAPI** for HTTP routing.
- **LiteLLM** as a unified client for multiple LLM providers.[web:32]
- **Celery + Redis** for asynchronous, long-running tasks.[web:49]
- **Ollama** for local models (health check and default summarizer).[web:44]

## Core modules

- `config.py`: Pydantic settings for host, port, models, Redis URLs.
- `schemas.py`: Pydantic request/response models mirroring the logical gRPC API.
- `celery_app.py`: Celery instance bound to Redis.
- `tasks.py`: Celery tasks implementing:
  - `embed_text_task`
  - `summarize_messages_task`
  - `extract_entities_task`
  - `score_importance_task`
- `main.py`: FastAPI app exposing:
  - `GET /health`
  - `POST /embed`
  - `POST /summarize`
  - `POST /extract_entities`
  - `POST /score_importance`
- `worker.py`: Celery worker entrypoint.

## Integration with Rust controller

The Rust `LlmBridgeClient` uses HTTP to call the bridge:

- `score_importance(&text, ..)` → `POST /score_importance`
- `summarize(messages, level, ..)` → `POST /summarize`
- (future) `embed(..)` → `POST /embed`
- (future) `extract_entities(..)` → `POST /extract_entities`

All heavy LLM calls live here, so the Rust core remains provider-agnostic.

## Error handling

- Validation errors → `400`
- Internal LLM / Celery errors → `500`
- Health endpoint degrades if Redis or Ollama are unavailable but still returns `200` with flags.
