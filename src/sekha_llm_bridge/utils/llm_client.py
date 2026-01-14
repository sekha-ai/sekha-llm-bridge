"""Unified LLM client using LiteLLM"""

import litellm
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from sekha_llm_bridge.config import settings

logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.set_verbose = False
litellm.drop_params = True  # Drop unsupported params


class LLMClient:
    """Unified interface for LLM operations"""

    def __init__(self):
        self.ollama_base_url = settings.ollama_base_url
        self.timeout = settings.ollama_timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_embedding(
        self, text: str, model: Optional[str] = None
    ) -> List[float]:
        """Generate embedding for text"""
        model = model or settings.embedding_model

        try:
            response = await litellm.aembedding(
                model=f"ollama/{model}",
                input=text,
                api_base=self.ollama_base_url,
                timeout=self.timeout,
            )

            return response.data[0]["embedding"]

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
        model = model or settings.summarization_model

        try:
            response = await litellm.acompletion(
                model=f"ollama/{model}",
                messages=messages,
                api_base=self.ollama_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama health and available models"""
        import httpx

        try:
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
                            settings.embedding_model in m["name"] for m in models
                        ),
                        "summarization_model_ready": any(
                            settings.summarization_model in m["name"] for m in models
                        ),
                    }

                return {"status": "unhealthy", "reason": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}


# Global client instance
llm_client = LLMClient()
