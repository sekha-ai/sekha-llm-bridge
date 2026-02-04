# Module 2: LLM Bridge Refactor - Complete ✅

## Overview
Module 2 implements the provider abstraction layer, model registry, routing logic, and new API endpoints for multi-provider LLM support.

## Completed Components

### 1. Provider Base Classes
**File:** `sekha-llm-bridge/src/sekha_llm_bridge/providers/base.py`

**Key Classes:**
- `LlmProvider` - Abstract base class for all providers
- `ChatMessage`, `ChatResponse` - Chat completion data models
- `EmbeddingResponse` - Embedding generation response
- `ModelInfo` - Model capability information
- `ProviderError`, `ProviderTimeoutError`, `ProviderRateLimitError`, `ProviderAuthError` - Exception hierarchy

**Features:**
- Abstract methods: `chat_completion()`, `chat_completion_stream()`, `generate_embedding()`, `list_models()`, `health_check()`
- Error handling with retry hints
- Vision and audio support flags

### 2. LiteLLM Provider Implementation
**File:** `sekha-llm-bridge/src/sekha_llm_bridge/providers/litellm_provider.py`

**Features:**
- Unified interface to all LLM providers via LiteLLM
- Supports: Ollama, OpenAI, Anthropic, OpenRouter, and 100+ providers
- Automatic model string formatting (`ollama/model`, `openai/model`)
- Vision support with image URLs or base64
- Streaming chat completions
- Health checks with latency measurement
- Error classification (timeout, rate limit, auth)

**Usage:**
```python
from sekha_llm_bridge.providers import LiteLlmProvider, ChatMessage

provider = LiteLlmProvider(
    provider_id="ollama_local",
    config={
        "provider_type": "ollama",
        "base_url": "http://localhost:11434",
        "timeout": 120,
    }
)

# Chat completion
response = await provider.chat_completion(
    messages=[ChatMessage(role="user", content="Hello!")],
    model="llama3.1:8b",
    temperature=0.7,
)

# Embedding
embedding = await provider.generate_embedding(
    text="Sample text",
    model="nomic-embed-text",
)
```

### 3. Cost Estimation
**File:** `sekha-llm-bridge/src/sekha_llm_bridge/pricing.py`

**Features:**
- Pricing table for 30+ models (OpenAI, Anthropic, Moonshot, DeepSeek, local)
- Cost estimation from token counts
- Cost estimation from text (approximate)
- Cost comparison across models
- Find cheapest model for a task

**Pricing Coverage:**
- ✅ OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5-turbo, embeddings)
- ✅ Anthropic (Claude 3 Opus, Sonnet, Haiku)
- ✅ Moonshot/Kimi (2.5, v1 variants)
- ✅ DeepSeek (v3, chat)
- ✅ Local models (Llama, Mistral, Mixtral - FREE)

**Usage:**
```python
from sekha_llm_bridge.pricing import estimate_cost, find_cheapest_model

# Estimate cost for a request
cost = estimate_cost(
    model_id="gpt-4o",
    input_tokens=1000,
    output_tokens=500,
)
print(f"Cost: ${cost:.4f}")  # $0.0125

# Find cheapest model
models = ["gpt-4o", "gpt-4o-mini", "llama3.1:8b"]
cheapest = find_cheapest_model(models, input_tokens=1000, output_tokens=500)
print(f"Cheapest: {cheapest}")  # llama3.1:8b (free!)
```

### 4. Model Registry
**File:** `sekha-llm-bridge/src/sekha_llm_bridge/registry.py`

**Responsibilities:**
- Track all configured providers
- Maintain model capability cache
- Route requests to optimal provider/model
- Handle fallback on provider failures
- Integrate circuit breakers
- Estimate costs

**Key Methods:**
- `route_with_fallback()` - Smart routing with automatic fallback
- `execute_with_circuit_breaker()` - Wrap operations with circuit breaker
- `get_provider_health()` - Get health status of all providers
- `list_all_models()` - List all available models

**Routing Logic:**
1. If preferred model specified, try it first
2. Find models matching task type
3. Filter by required capabilities (vision, audio)
4. Check circuit breaker states (skip open circuits)
5. Estimate costs and filter by budget
6. Sort by provider priority
7. Return best match with fallback list

**Usage:**
```python
from sekha_llm_bridge.registry import registry
from sekha_llm_bridge.config import ModelTask

# Route a request
result = await registry.route_with_fallback(
    task=ModelTask.CHAT_SMART,
    preferred_model="gpt-4o",
    max_cost=0.10,
)

print(f"Using: {result.provider.provider_id}/{result.model_id}")
print(f"Estimated cost: ${result.estimated_cost:.4f}")
print(f"Reason: {result.reason}")

# Execute with circuit breaker protection
response = await registry.execute_with_circuit_breaker(
    provider_id=result.provider.provider_id,
    operation=result.provider.chat_completion,
    messages=[...],
    model=result.model_id,
)
```

### 5. V2.0 API Endpoints
**File:** `sekha-llm-bridge/src/sekha_llm_bridge/routes_v2.py`

**New Endpoints:**

#### `GET /api/v1/models`
List all available models across all providers.

```bash
curl http://localhost:5001/api/v1/models
```

**Response:**
```json
[
  {
    "model_id": "nomic-embed-text",
    "provider_id": "ollama_local",
    "task": "embedding",
    "context_window": 512,
    "dimension": 768,
    "supports_vision": false,
    "supports_audio": false
  },
  {
    "model_id": "gpt-4o",
    "provider_id": "openai_cloud",
    "task": "chat_smart",
    "context_window": 128000,
    "dimension": null,
    "supports_vision": true,
    "supports_audio": false
  }
]
```

#### `POST /api/v1/route`
Route a request to the optimal provider and model.

```bash
curl -X POST http://localhost:5001/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "task": "chat_smart",
    "require_vision": false,
    "max_cost": 0.10
  }'
```

**Response:**
```json
{
  "provider_id": "ollama_local",
  "model_id": "llama3.1:8b",
  "estimated_cost": 0.0,
  "reason": "Selected by priority (1)",
  "provider_type": "ollama"
}
```

#### `GET /api/v1/health/providers`
Get health status of all providers.

```bash
curl http://localhost:5001/api/v1/health/providers
```

**Response:**
```json
{
  "providers": {
    "ollama_local": {
      "provider_type": "ollama",
      "circuit_breaker": {
        "state": "closed",
        "failure_count": 0,
        "success_count": 15,
        "last_failure_time": null
      },
      "models_count": 3
    },
    "openai_cloud": {
      "provider_type": "openai",
      "circuit_breaker": {
        "state": "open",
        "failure_count": 5,
        "success_count": 0,
        "last_failure_time": "2026-02-04T22:30:00Z"
      },
      "models_count": 2
    }
  },
  "total_providers": 2,
  "healthy_providers": 1,
  "total_models": 5
}
```

#### `GET /api/v1/tasks`
List all available task types.

```bash
curl http://localhost:5001/api/v1/tasks
```

**Response:**
```json
[
  "embedding",
  "chat_small",
  "chat_large",
  "chat_smart",
  "vision",
  "audio"
]
```

### 6. Updated Main App
**File:** `sekha-llm-bridge/src/sekha_llm_bridge/main.py`

**Changes:**
- ✅ Include v2.0 router endpoints
- ✅ Initialize registry on startup
- ✅ Check provider health at startup
- ✅ Updated version to 2.0.0
- ✅ Backward compatible with existing endpoints

## Testing Module 2

### 1. Start Bridge with V2.0 Config

```bash
cd sekha-llm-bridge

# Use example config
export SEKHA__LLM_PROVIDERS='[
  {
    "id": "ollama_local",
    "type": "ollama",
    "base_url": "http://localhost:11434",
    "priority": 1,
    "models": [
      {"model_id": "nomic-embed-text", "task": "embedding", "context_window": 512, "dimension": 768},
      {"model_id": "llama3.1:8b", "task": "chat_small", "context_window": 8192},
      {"model_id": "llama3.1:8b", "task": "chat_smart", "context_window": 8192}
    ]
  }
]'
export SEKHA__DEFAULT_MODELS='{"embedding": "nomic-embed-text", "chat_fast": "llama3.1:8b", "chat_smart": "llama3.1:8b"}'

python -m sekha_llm_bridge.main
```

### 2. Test New Endpoints

```bash
# List all models
curl http://localhost:5001/api/v1/models | jq

# Request routing
curl -X POST http://localhost:5001/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task": "embedding"}' | jq

# Provider health
curl http://localhost:5001/api/v1/health/providers | jq

# List tasks
curl http://localhost:5001/api/v1/tasks | jq
```

### 3. Test Cost Estimation

```python
from sekha_llm_bridge.pricing import (
    estimate_cost,
    compare_costs,
    find_cheapest_model,
)

# Estimate GPT-4o cost
cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
print(f"GPT-4o: ${cost:.4f}")  # $0.0125

# Compare multiple models
models = ["gpt-4o", "gpt-4o-mini", "claude-3-sonnet", "llama3.1:8b"]
costs = compare_costs(models, input_tokens=1000, output_tokens=500)
for model, cost in costs.items():
    print(f"{model}: ${cost:.4f}")

# Find cheapest under budget
cheapest = find_cheapest_model(models, 1000, 500, max_cost=0.01)
print(f"Cheapest under $0.01: {cheapest}")
```

### 4. Test Provider with Circuit Breaker

```python
from sekha_llm_bridge.registry import registry
from sekha_llm_bridge.config import ModelTask
from sekha_llm_bridge.providers.base import ChatMessage, MessageRole

# Route request
result = await registry.route_with_fallback(
    task=ModelTask.CHAT_SMALL,
    max_cost=0.05,
)

print(f"Routed to: {result.provider.provider_id}/{result.model_id}")

# Execute with circuit breaker
messages = [
    ChatMessage(role=MessageRole.USER, content="Hello!")
]

response = await registry.execute_with_circuit_breaker(
    provider_id=result.provider.provider_id,
    operation=result.provider.chat_completion,
    messages=messages,
    model=result.model_id,
)

print(f"Response: {response.content}")
print(f"Tokens: {response.usage['total_tokens']}")
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI App                       │
│  ┌──────────────────────────────────────────────┐  │
│  │         V2 Router (routes_v2.py)             │  │
│  │  /api/v1/models                              │  │
│  │  /api/v1/route                               │  │
│  │  /api/v1/health/providers                    │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼───────────────────────────┐  │
│  │       Model Registry (registry.py)           │  │
│  │  - Provider tracking                         │  │
│  │  - Model cache                               │  │
│  │  - Routing logic                             │  │
│  │  - Circuit breakers                          │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼───────────────────────────┐  │
│  │    Providers (providers/)                    │  │
│  │  ┌────────────────────────────────────────┐  │  │
│  │  │  LlmProvider (base.py)                 │  │  │
│  │  │  - Abstract interface                  │  │  │
│  │  └────────────────┬───────────────────────┘  │  │
│  │                   │                           │  │
│  │  ┌────────────────▼───────────────────────┐  │  │
│  │  │  LiteLlmProvider (litellm_provider.py) │  │  │
│  │  │  - Unified LLM access                  │  │  │
│  │  │  - Ollama, OpenAI, Anthropic, etc.     │  │  │
│  │  └────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │       Cost Estimation (pricing.py)           │  │
│  │  - Token-based pricing                       │  │
│  │  - Cost comparison                           │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Completion Checklist

- [x] Provider abstract base class (base.py)
- [x] LiteLLM provider implementation
- [x] Cost estimation module with pricing table
- [x] Model registry with routing logic
- [x] Circuit breaker integration
- [x] V2.0 API endpoints (/models, /route, /health/providers)
- [x] Update main.py to include v2 routes
- [x] Provider health checks
- [x] Vision support in messages
- [x] Streaming support
- [x] Error classification (timeout, rate limit, auth)
- [x] Documentation complete

## Key Features Delivered

✅ **Provider Abstraction** - Clean interface for any LLM provider  
✅ **Multi-Provider Support** - Ollama, OpenAI, Anthropic, OpenRouter via LiteLLM  
✅ **Smart Routing** - Task-based, cost-aware, capability-aware  
✅ **Automatic Fallback** - Circuit breakers with provider fallback  
✅ **Cost Estimation** - Real pricing for 30+ models  
✅ **Vision Support** - Images in chat messages  
✅ **Streaming** - Chat completion streaming  
✅ **Health Monitoring** - Provider health and circuit breaker states  

## Files Modified in Module 2

```
sekha-llm-bridge/
├── src/sekha_llm_bridge/
│   ├── providers/
│   │   ├── __init__.py (NEW - 283 B)
│   │   ├── base.py (NEW - 6.7 KB)
│   │   └── litellm_provider.py (NEW - 13.0 KB)
│   ├── pricing.py (NEW - 7.5 KB)
│   ├── registry.py (NEW - 12.4 KB)
│   ├── routes_v2.py (NEW - 5.4 KB)
│   └── main.py (UPDATED - 12.7 KB)
└── docs/
    └── MODULE_2_README.md (NEW)
```

---

**Module 2 Status:** ✅ **COMPLETE**  
**Estimated Time:** 5-6 days → **Actual: Completed in same session**  
**Ready for Module 3:** Yes  
**Backward Compatible:** Yes ✅  
**New Endpoints:** 4 (/models, /route, /health/providers, /tasks)
