"""Unified LLM client using v2.0 Provider Registry."""

import logging
from typing import Any, Dict, List, Optional

from sekha_llm_bridge.config import ModelTask, settings
from sekha_llm_bridge.registry import registry

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified interface for LLM operations using provider registry."""

    def __init__(self):
        """Initialize LLM client with provider registry."""
        self.registry = registry

    async def generate_embedding(
        self, text: str, model: Optional[str] = None
    ) -> List[float]:
        """Generate embedding for text using provider registry.

        Args:
            text: Text to embed
            model: Optional specific model to use

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If no suitable provider available
        """
        try:
            # Route to appropriate provider
            routing_result = await self.registry.route_with_fallback(
                task=ModelTask.EMBEDDING,
                preferred_model=model,
            )

            # Execute embedding request
            result = await self.registry.execute_with_circuit_breaker(
                provider_id=routing_result.provider.provider_id,
                operation=routing_result.provider.generate_embedding,
                text=text,
                model=routing_result.model_id,
            )

            return result.embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        require_vision: bool = False,
    ) -> str:
        """Generate LLM completion using provider registry.

        Args:
            messages: Chat messages
            model: Optional specific model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            require_vision: Whether vision support is required

        Returns:
            Generated text content

        Raises:
            RuntimeError: If no suitable provider available
        """
        try:
            # Determine task type based on model preference
            # Default to chat_smart if no model specified
            task = ModelTask.CHAT_SMART
            if model:
                # Check if model name suggests a specific task
                model_lower = model.lower()
                if "mini" in model_lower or "small" in model_lower:
                    task = ModelTask.CHAT_SMALL
                elif "large" in model_lower:
                    task = ModelTask.CHAT_LARGE

            # Route to appropriate provider
            routing_result = await self.registry.route_with_fallback(
                task=task,
                preferred_model=model,
                require_vision=require_vision,
            )

            # Convert messages to ChatMessage format
            from sekha_llm_bridge.providers.base import ChatMessage, MessageRole

            chat_messages = [
                ChatMessage(
                    role=MessageRole(msg["role"]), content=msg["content"]
                )
                for msg in messages
            ]

            # Execute completion request
            result = await self.registry.execute_with_circuit_breaker(
                provider_id=routing_result.provider.provider_id,
                operation=routing_result.provider.chat_completion,
                messages=chat_messages,
                model=routing_result.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return result.content

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all registered providers.

        Returns:
            Dictionary with provider health status
        """
        try:
            provider_health = self.registry.get_provider_health()
            models = self.registry.list_all_models()

            # Check if we have essential capabilities
            has_embedding = any(m["task"] == "embedding" for m in models)
            has_chat = any(
                m["task"] in ["chat_small", "chat_smart", "chat_large"] for m in models
            )

            status = "healthy" if (has_embedding and has_chat) else "degraded"

            return {
                "status": status,
                "providers": provider_health,
                "models_count": len(models),
                "has_embedding": has_embedding,
                "has_chat": has_chat,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "reason": str(e),
            }


# Global client instance
llm_client = LLMClient()
