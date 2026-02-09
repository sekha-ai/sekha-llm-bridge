# Module 5: Vision Pass-Through & Documentation

**Status**: ✅ Complete  
**Version**: 2.0.0  
**Date**: February 2026

## Overview

Module 5 implements vision support for multi-modal AI interactions and provides comprehensive documentation for the v2.0 release.

### Key Features

1. **Vision Pass-Through**: Multi-modal message support (text + images)
2. **Documentation**: Complete v2.0 release documentation
3. **Migration Guide**: v1.x → v2.0 upgrade instructions

---

## Vision Support

### Architecture

Vision requests flow through the stack:

```
Client → Controller (MessageDto) → Bridge (ChatMessage) → Provider (LiteLLM) → Vision Model
```

**Components:**
- **Controller**: Accepts multi-modal messages via REST API
- **Bridge**: Routes to vision-capable models
- **LiteLLM**: Converts to provider-specific formats
- **Models**: GPT-4o, Claude 3, Gemini Pro Vision, etc.

### Supported Image Formats

#### 1. Image URLs
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in this image?"},
    {
      "type": "image_url",
      "image_url": {"url": "https://example.com/photo.jpg"}
    }
  ]
}
```

#### 2. Base64 Encoded Images
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image"},
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
      }
    }
  ]
}
```

#### 3. Multiple Images
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Compare these two images"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
    {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
  ]
}
```

### Vision-Capable Models

Configured in `config/llm_providers.yaml`:

```yaml
providers:
  - id: openai_cloud
    provider_type: openai
    api_key: ${OPENAI_API_KEY}
    models:
      - model_id: gpt-4o
        task: chat_smart
        context_window: 128000
        supports_vision: true  # ← Enable vision
      
      - model_id: gpt-4o-mini
        task: chat_small
        context_window: 128000
        supports_vision: true

  - id: anthropic_cloud
    provider_type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    models:
      - model_id: claude-3-opus
        task: chat_smart
        context_window: 200000
        supports_vision: true
      
      - model_id: claude-3-sonnet
        task: chat_small
        context_window: 200000
        supports_vision: true
```

---

## Usage Examples

### Example 1: Simple Image Analysis

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/photo.jpg"
                        }
                    }
                ]
            }
        ],
        "require_vision": True,  # Route to vision-capable model
        "max_cost": 0.05        # Max $0.05 per request
    }
)

print(response.json()['choices'][0]['message']['content'])
```

### Example 2: Base64 Image Upload

```python
import base64
import requests

# Read and encode image
with open("photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        "require_vision": True
    }
)
```

### Example 3: Multi-Turn Vision Conversation

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
        ]
    }
]

# First request
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={"messages": messages, "require_vision": True}
)

# Add assistant response
messages.append({
    "role": "assistant",
    "content": response.json()['choices'][0]['message']['content']
})

# Follow-up question (no new image needed)
messages.append({
    "role": "user",
    "content": "What colors are prominent?"
})

# Second request
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={"messages": messages, "require_vision": True}
)
```

### Example 4: Compare Multiple Images

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images. What are the differences?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/before.jpg"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/after.jpg"}}
                ]
            }
        ],
        "require_vision": True,
        "preferred_model": "gpt-4o"  # Best for detailed comparison
    }
)
```

---

## Vision Routing

### Automatic Model Selection

When `require_vision: true`, the bridge:

1. **Filters** to vision-capable models only
2. **Sorts** by priority (cost, availability)
3. **Selects** optimal model with fallback
4. **Routes** request to chosen provider

```python
# Automatic routing to vision model
response = requests.post("/v1/chat/completions", json={
    "messages": [vision_message],
    "require_vision": True  # Bridge picks best vision model
})
```

### Explicit Model Selection

```python
# Force specific vision model
response = requests.post("/v1/chat/completions", json={
    "model": "gpt-4o",
    "messages": [vision_message],
    "require_vision": True
})
```

### Cost-Aware Vision Routing

```python
# Prefer cheaper vision models
response = requests.post("/v1/chat/completions", json={
    "messages": [vision_message],
    "require_vision": True,
    "max_cost": 0.01  # Will use gpt-4o-mini over gpt-4o
})
```

---

## Image Requirements

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif) - static and animated
- WebP (.webp)

### Size Limits
- **URL Images**: Up to provider limit (typically 20MB)
- **Base64 Images**: 
  - Recommended: < 5MB
  - Maximum: 20MB (varies by provider)

### Resolution Recommendations
- **Minimum**: 512x512 pixels
- **Optimal**: 1024x1024 to 2048x2048
- **Maximum**: 4096x4096 (model dependent)

### Detail Levels

Some models support detail specification:

```json
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/photo.jpg",
    "detail": "high"  // "low", "high", or "auto"
  }
}
```

- **low**: Faster, cheaper, less detail (~85 tokens)
- **high**: Slower, costlier, more detail (~255-765 tokens)
- **auto**: Model decides based on image

---

## Best Practices

### 1. Image Optimization

```python
from PIL import Image
import io
import base64

def optimize_image(image_path, max_size=1024):
    """Resize and compress image for vision API."""
    img = Image.open(image_path)
    
    # Resize if too large
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    # Compress to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85, optimize=True)
    
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
```

### 2. Error Handling

```python
try:
    response = requests.post("/v1/chat/completions", json={
        "messages": [vision_message],
        "require_vision": True
    })
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        # Invalid image format or size
        print("Image error:", e.response.json())
    elif e.response.status_code == 503:
        # No vision models available
        print("No vision models available")
    else:
        raise
```

### 3. Cost Management

Vision requests are typically more expensive:

```python
# Estimate cost before request
route_response = requests.post("/api/v1/route", json={
    "task": "chat_smart",
    "require_vision": True
})

estimated_cost = route_response.json()['estimated_cost']
print(f"Estimated cost: ${estimated_cost:.4f}")

if estimated_cost < 0.05:  # Budget check
    # Make actual request
    response = requests.post("/v1/chat/completions", ...)
```

### 4. Caching Images

For repeated analysis of the same image:

```python
# Send image once, ask multiple questions
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
        ]
    }
]

# First question
response1 = requests.post("/v1/chat/completions", json={"messages": messages})
messages.append({"role": "assistant", "content": response1.json()['choices'][0]['message']['content']})

# Follow-up questions (image already in context)
messages.append({"role": "user", "content": "What colors are present?"})
response2 = requests.post("/v1/chat/completions", json={"messages": messages})
```

---

## Troubleshooting

### No Vision Models Available

**Error**: `RuntimeError: No suitable provider available for task 'chat_smart' (vision=True)`

**Solution**:
1. Check `config/llm_providers.yaml` has models with `supports_vision: true`
2. Verify provider API keys are set
3. Check provider health: `curl http://localhost:8000/api/v1/health`

### Image Too Large

**Error**: `400 Bad Request: Image size exceeds limit`

**Solution**:
```python
# Resize image before sending
from PIL import Image
img = Image.open("large_photo.jpg")
img.thumbnail((2048, 2048))
img.save("resized_photo.jpg", optimize=True, quality=85)
```

### Invalid Image Format

**Error**: `400 Bad Request: Unsupported image format`

**Solution**:
```python
# Convert to JPEG
from PIL import Image
img = Image.open("image.webp")
img.convert('RGB').save("image.jpg")
```

### High Costs

**Issue**: Vision requests are expensive

**Solution**:
```python
# Use detail="low" for simple questions
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/photo.jpg",
    "detail": "low"  # Reduces token usage ~70%
  }
}

# Or set cost limits
response = requests.post("/v1/chat/completions", json={
    "messages": [vision_message],
    "max_cost": 0.02  # Max $0.02 per request
})
```

---

## Testing

Run vision tests:

```bash
# Unit tests
pytest tests/test_vision.py -v

# Integration tests
pytest tests/integration/test_provider_routing.py::TestVisionRouting -v

# E2E tests
pytest tests/test_e2e_stack.py::TestProxyBridgeIntegration::test_vision_request_routing -v
```

---

## Implementation Details

### Controller (Rust)

**File**: `src/api/dto.rs`

```rust
pub struct MessageDto {
    pub role: String,
    pub content: MessageContent,  // Text or Parts
}

pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}
```

### Bridge (Python)

**File**: `src/sekha_llm_bridge/providers/base.py`

```python
@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    images: Optional[List[str]] = None  # URLs or base64
```

**File**: `src/sekha_llm_bridge/providers/litellm_provider.py`

```python
def _convert_messages(self, messages: List[ChatMessage]):
    for msg in messages:
        if msg.images:
            content_parts = [{"type": "text", "text": msg.content}]
            for image in msg.images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image}
                })
            message_dict["content"] = content_parts
```

---

## API Reference

### Vision Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | Array | Yes | Message array with content parts |
| `require_vision` | Boolean | No | Force vision-capable model |
| `max_cost` | Float | No | Maximum cost per request |
| `preferred_model` | String | No | Preferred vision model |
| `detail` | String | No | Image detail level (low/high/auto) |

### Content Part Types

**Text Part**:
```json
{"type": "text", "text": "Your question here"}
```

**Image URL Part**:
```json
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/image.jpg",
    "detail": "auto"  // optional
  }
}
```

---

## Related Documentation

- [CHANGELOG.md](../CHANGELOG.md) - v2.0 release notes
- [MIGRATION.md](../MIGRATION.md) - Upgrade guide from v1.x
- [Multi-Provider Setup](./MULTI_PROVIDER.md) - Provider configuration
- [Cost Management](./COST_MANAGEMENT.md) - Budget and cost controls

---

## Summary

Module 5 adds comprehensive vision support to Sekha:

✅ **Multi-modal messages** - Text + images in same request  
✅ **Automatic routing** - Finds best vision model  
✅ **Cost awareness** - Budget controls for expensive vision requests  
✅ **Multiple formats** - URLs and base64 encoding  
✅ **Provider agnostic** - Works with OpenAI, Anthropic, Google, etc.  
✅ **Full documentation** - CHANGELOG, MIGRATION, and usage guides  

Vision is now a first-class feature in Sekha v2.0!
