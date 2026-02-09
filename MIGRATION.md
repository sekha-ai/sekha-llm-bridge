# Migration Guide: v1.x → v2.0

This guide helps you upgrade from Sekha LLM Bridge v1.x to v2.0.

## Overview of Changes

v2.0 introduces a **multi-provider registry architecture** with significant improvements:

- ✅ Multiple LLM provider support (Ollama, OpenAI, Anthropic, etc.)
- ✅ Intelligent request routing with cost awareness
- ✅ Circuit breakers and automatic failover
- ✅ Vision support (multi-modal messages)
- ✅ Multi-dimension embedding collections
- ⚠️ Breaking changes to configuration format
- ⚠️ API enhancements (backward compatible with caveats)

---

## Breaking Changes

### 1. Configuration Format ⚠️ **BREAKING**

**Before (v1.x)**: Environment variables only
```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
CHAT_MODEL=llama3.1:8b
```

**After (v2.0)**: YAML configuration file
```yaml
# config/llm_providers.yaml
providers:
  - id: ollama_local
    provider_type: ollama
    base_url: http://localhost:11434
    priority: 1
    models:
      - model_id: nomic-embed-text
        task: embedding
        context_window: 8192
        dimension: 768
      
      - model_id: llama3.1:8b
        task: chat_small
        context_window: 8192
      
      - model_id: llama3.1:70b
        task: chat_smart
        context_window: 8192

default_models:
  embedding: nomic-embed-text
  chat_fast: llama3.1:8b
  chat_smart: llama3.1:70b

routing:
  cost_aware: true
  max_cost_per_request: 0.10  # $0.10 per request
  circuit_breaker:
    failure_threshold: 5
    timeout_secs: 60
    success_threshold: 2
```

**Migration Steps:**

1. Create `config/llm_providers.yaml` in your project root
2. Copy the example above and adjust for your setup
3. Set environment variable: `LLM_CONFIG_PATH=config/llm_providers.yaml`
4. Remove old environment variables (they're now ignored)

---

### 2. Embedding Collection Names ⚠️ **BREAKING**

**Before (v1.x)**: Single collection per type
```
conversations
query_results
```

**After (v2.0)**: Dimension-specific collections
```
conversations_768    (for 768-dimensional embeddings)
conversations_1536   (for 1536-dimensional embeddings)
conversations_3072   (for 3072-dimensional embeddings)
```

**Why?** Supports switching embedding models with different dimensions.

**Migration Options:**

**Option A: Automatic (Recommended)**
The bridge will automatically:
1. Detect existing collections without suffix
2. Determine their dimension from first embedding
3. Rename to `{name}_{dimension}`
4. Future embeddings use dimension-specific collections

**Option B: Manual Migration**
```python
# migration_script.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_data")

# Get old collection
old_collection = client.get_collection("conversations")

# Get all data
results = old_collection.get(include=["embeddings", "documents", "metadatas"])

# Detect dimension
if results['embeddings']:
    dimension = len(results['embeddings'][0])
    print(f"Detected dimension: {dimension}")
    
    # Create new collection
    new_collection = client.get_or_create_collection(f"conversations_{dimension}")
    
    # Copy data
    new_collection.add(
        ids=results['ids'],
        embeddings=results['embeddings'],
        documents=results['documents'],
        metadatas=results['metadatas']
    )
    
    print(f"Migrated {len(results['ids'])} documents")
    
    # Optional: Delete old collection
    # client.delete_collection("conversations")
```

**Option C: Start Fresh**
If your existing data isn't critical:
1. Stop the bridge
2. Delete `./chroma_data` directory
3. Start v2.0 - collections will be created automatically

---

### 3. API Request Changes

#### Chat Completions

**Before (v1.x)**:
```python
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello"}]
})
```

**After (v2.0)**: Same format works, but new options available
```python
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llama3.1:8b",  # Optional: omit for auto-routing
    "messages": [{"role": "user", "content": "Hello"}],
    # New optional parameters:
    "require_vision": False,
    "max_cost": 0.01,  # Max $0.01 per request
    "preferred_model": "llama3.1:8b"  # Prefer this, fallback if unavailable
})
```

#### Vision Messages (New in v2.0)

```python
# Text + Image
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gpt-4o",  # Vision-capable model
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    "require_vision": True
})
```

#### Routing API (New in v2.0)

```python
# Get optimal provider/model without making request
response = requests.post("http://localhost:8000/api/v1/route", json={
    "task": "chat_smart",
    "require_vision": False,
    "max_cost": 0.05
})

print(response.json())
# {
#   "provider_id": "ollama_local",
#   "model_id": "llama3.1:70b",
#   "estimated_cost": 0.0,
#   "reason": "Local model available, free"
# }
```

---

## New Features (Non-Breaking)

### Multiple Providers

Add OpenAI as a fallback:

```yaml
providers:
  # Primary: Local Ollama (free)
  - id: ollama_local
    provider_type: ollama
    base_url: http://localhost:11434
    priority: 1
    models:
      - model_id: llama3.1:8b
        task: chat_small
        context_window: 8192
  
  # Fallback: OpenAI (paid)
  - id: openai_cloud
    provider_type: openai
    api_key: ${OPENAI_API_KEY}
    priority: 2
    models:
      - model_id: gpt-4o-mini
        task: chat_small
        context_window: 128000
        supports_vision: true
      
      - model_id: gpt-4o
        task: chat_smart
        context_window: 128000
        supports_vision: true
```

**Behavior**: 
- Requests try `ollama_local` first (priority 1)
- If Ollama is down, automatically fallback to `openai_cloud` (priority 2)
- Circuit breaker prevents repeated calls to failing providers

### Cost Limits

```python
# Hard limit per request
response = requests.post("/v1/chat/completions", json={
    "messages": [...],
    "max_cost": 0.01  # Reject if estimated cost > $0.01
})

# Bridge will:
# 1. Estimate cost for each candidate model
# 2. Skip models exceeding max_cost
# 3. Prefer cheaper models when equivalent
# 4. Return error if no affordable models available
```

### Circuit Breakers

Automatic protection against failing providers:

```yaml
routing:
  circuit_breaker:
    failure_threshold: 5      # Open after 5 failures
    timeout_secs: 60         # Try again after 60 seconds
    success_threshold: 2     # Close after 2 successes
```

Check circuit breaker status:
```python
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json()['providers'])
# {
#   "ollama_local": {
#     "circuit_breaker": {
#       "state": "closed",  # or "open", "half_open"
#       "failure_count": 0,
#       "last_failure": null
#     }
#   }
# }
```

---

## Code Changes

### Python Client

**Before (v1.x)**:
```python
from sekha_llm_bridge import LlmBridge

bridge = LlmBridge(base_url="http://localhost:8000")
response = bridge.chat("Hello", model="llama3.1:8b")
```

**After (v2.0)**: Same API, enhanced internally
```python
from sekha_llm_bridge import LlmBridge

bridge = LlmBridge(base_url="http://localhost:8000")

# Auto-routing (uses cheapest available model)
response = bridge.chat("Hello")

# Preferred model with fallback
response = bridge.chat("Hello", preferred_model="llama3.1:8b")

# Vision request
response = bridge.chat_with_image(
    "What's in this image?",
    image_url="https://example.com/image.jpg"
)

# Cost-aware request
response = bridge.chat("Hello", max_cost=0.01)
```

### Health Checks

**Before (v1.x)**:
```python
response = requests.get("http://localhost:8000/health")
# {"status": "healthy", "version": "0.1.0"}
```

**After (v2.0)**: Includes provider details
```python
response = requests.get("http://localhost:8000/api/v1/health")
# {
#   "status": "healthy",
#   "version": "2.0.0",
#   "providers": {
#     "ollama_local": {
#       "status": "healthy",
#       "models_count": 3,
#       "circuit_breaker": {"state": "closed"},
#       "latency_ms": 45
#     },
#     "openai_cloud": {
#       "status": "healthy",
#       "models_count": 2,
#       "circuit_breaker": {"state": "closed"},
#       "latency_ms": 120
#     }
#   }
# }
```

---

## Compatibility Notes

### ✅ Backward Compatible

- Old API endpoints still work (`/v1/chat/completions`, `/v1/embeddings`)
- Simple text messages work without changes
- Model parameter is now optional (auto-routing)
- Health endpoint at `/health` redirects to `/api/v1/health`

### ⚠️ May Require Updates

- **Configuration**: Must migrate to YAML format
- **Embedding collections**: Will be auto-migrated with dimension suffix
- **Health response**: Structure changed (backward compatible at top level)
- **Error responses**: More detailed, may need client updates

### ❌ No Longer Supported

- Direct environment variable configuration (use YAML)
- Single-provider mode (define one provider in YAML)
- Non-dimensional collection names (auto-migrated)

---

## Testing Your Migration

### 1. Verify Configuration
```bash
# Test config loading
python -c "from sekha_llm_bridge.config import settings; print(settings.providers)"
```

### 2. Check Provider Health
```bash
curl http://localhost:8000/api/v1/health | jq
```

### 3. Test Routing
```bash
curl -X POST http://localhost:8000/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task": "chat_small"}' | jq
```

### 4. Test Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}]
  }' | jq
```

### 5. Verify Embeddings
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Test embedding"}' | jq
```

---

## Rollback Plan

If you encounter issues:

### Quick Rollback
```bash
# 1. Stop v2.0
docker-compose down

# 2. Checkout v1.x
git checkout v0.1.0

# 3. Restore old .env
cp .env.backup .env

# 4. Start v1.x
docker-compose up -d
```

### Data Rollback
If you migrated embeddings manually:
```python
# Restore from backup
import shutil
shutil.copytree("./chroma_data.backup", "./chroma_data", dirs_exist_ok=True)
```

---

## Getting Help

- **GitHub Issues**: https://github.com/sekha-ai/sekha-llm-bridge/issues
- **Discussions**: https://github.com/sekha-ai/sekha-llm-bridge/discussions
- **Documentation**: https://sekha-ai.github.io/sekha-llm-bridge/

---

## Summary Checklist

- [ ] Create `config/llm_providers.yaml` with your providers
- [ ] Set `LLM_CONFIG_PATH` environment variable
- [ ] Test configuration loading
- [ ] Verify provider health checks pass
- [ ] Test chat completion with auto-routing
- [ ] Test embeddings (collections auto-migrate)
- [ ] Update client code for new features (optional)
- [ ] Monitor circuit breaker states
- [ ] Set cost limits if using paid providers
- [ ] Remove old environment variables from `.env`

**Estimated migration time**: 15-30 minutes for basic setup, 1-2 hours for multi-provider configuration.
