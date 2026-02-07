"""Unified LLM client using LiteLLM

NOTE: This is legacy code that should be refactored to use the provider registry.
For now, using hardcoded defaults where settings don't match v2.0 config.
"""

import logging
import os
from typing import Any, Dict, List, Optional, cast

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from sekha_llm_bridge.config import get_settings

logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.set_verbose = False
litellm.drop_params = True  # Drop unsupported params


class LLMClient:
    """Unified interface for LLM operations"""

    def __init__(self):
        # TODO: Refactor to use provider registry
        self.ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = 120

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_embedding(
        self, text: str, model: Optional[str] = None
    ) -> List[float]:
        """Generate embedding for text"""
        if model is None:
            settings = get_settings()
            model = settings.default_models.embedding or "nomic-embed-text"

        try:
            response = await litellm.aembedding(
                model=f"ollama/{model}",
                input=text,
                api_base=self.ollama_base_url,
                timeout=self.timeout,
            )

            embedding = cast(List[float], response.data[0]["embedding"])
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate LLM completion"""
        if model is None:
            settings = get_settings()
            model = settings.default_models.chat_smart or "llama3.1:8b"

        try:
            response = await litellm.acompletion(
                model=f"ollama/{model}",
                messages=messages,
                api_base=self.ollama_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )

            content = cast(str, response.choices[0].message.content)
            return content

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama health and available models"""
        import httpx

        try:
            settings = get_settings()
            embedding_model = settings.default_models.embedding or "nomic-embed-text"
            chat_model = settings.default_models.chat_smart or "llama3.1:8b"

            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.ollama_base_url}/api/tags")

                if response.status_code == 200:
                    models = response.json().get("models", [])

                    return {
                        "status": "healthy",
                        "ollama_url": self.ollama_base_url,
                        "models_available": [m["name"] for m in models],
                        "embedding_model_ready": any(
                            embedding_model in m["name"] for m in models
                        ),
                        "summarization_model_ready": any(
                            chat_model in m["name"] for m in models
                        ),
                    }

                return {"status": "unhealthy", "reason": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}


# Global client instance
llm_client = LLMClient()
