"""LiteLLM provider implementation for unified LLM access."""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, cast

import litellm

from .base import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    LlmProvider,
    MessageRole,
    ModelInfo,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


class LiteLlmProvider(LlmProvider):
    """Provider implementation using LiteLLM for unified access.

    LiteLLM provides a unified interface to multiple LLM providers:
    - Ollama (ollama/model-name)
    - OpenAI (openai/model-name or just model-name)
    - Anthropic (anthropic/model-name)
    - OpenRouter (openrouter/model-name)
    - And many more
    """

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        """Initialize LiteLLM provider.

        Args:
            provider_id: Unique identifier for this provider
            config: Configuration including:
                - provider_type: "ollama", "openai", "anthropic", etc.
                - base_url: API base URL
                - api_key: Optional API key
                - timeout: Request timeout in seconds
                - models: List of model configurations
        """
        super().__init__(provider_id, config)

        self._provider_type: str = cast(str, config.get("provider_type", "ollama"))
        self.base_url = config.get("base_url")
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 120)

        # Configure LiteLLM
        if self.api_key:
            # Set API key in environment for LiteLLM
            key_name = self._get_api_key_env_name()
            if key_name:
                import os

                os.environ[key_name] = self.api_key

        logger.info(
            f"Initialized LiteLLM provider '{provider_id}' "
            f"(type: {self._provider_type}, base_url: {self.base_url})"
        )

    @property
    def provider_type(self) -> str:
        return self._provider_type

    def _get_api_key_env_name(self) -> Optional[str]:
        """Get the environment variable name for the API key."""
        mapping: Dict[str, str] = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        return cast(Optional[str], mapping.get(self._provider_type))

    def get_model_string(self, model: str) -> str:
        """Convert model ID to LiteLLM format.

        Args:
            model: Model identifier (e.g., 'llama3.1:8b', 'gpt-4o')

        Returns:
            LiteLLM model string (e.g., 'ollama/llama3.1:8b', 'gpt-4o')
        """
        # If already prefixed, return as-is
        if "/" in model:
            return model

        # For Ollama, prefix with 'ollama/'
        if self._provider_type == "ollama":
            return f"ollama/{model}"

        # For OpenRouter, prefix with 'openrouter/'
        if self._provider_type == "openrouter":
            return f"openrouter/{model}"

        # For OpenAI and Anthropic, LiteLLM handles it
        return model

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Generate a chat completion using LiteLLM."""
        try:
            # Convert messages to LiteLLM format
            litellm_messages = self._convert_messages(messages)

            # Get full model string
            full_model = self.get_model_string(model)

            # Prepare kwargs
            call_kwargs = {
                "model": full_model,
                "messages": litellm_messages,
                "temperature": temperature,
                "timeout": self.timeout,
            }

            if max_tokens:
                call_kwargs["max_tokens"] = max_tokens

            if self.base_url and self._provider_type == "ollama":
                call_kwargs["api_base"] = self.base_url

            # Add any additional kwargs
            call_kwargs.update(kwargs)

            # Make request
            logger.debug(f"Calling LiteLLM with model: {full_model}")
            response = await litellm.acompletion(**call_kwargs)

            # Extract response
            choice = response.choices[0]
            content = choice.message.content or ""

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return ChatResponse(
                content=content,
                model=model,
                provider_id=self.provider_id,
                finish_reason=choice.finish_reason,
                usage=usage,
                metadata={"litellm_model": full_model},
            )

        except asyncio.TimeoutError as e:
            raise ProviderTimeoutError(
                f"Request to {self.provider_id} timed out after {self.timeout}s",
                self.provider_id,
            ) from e
        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limit
            if "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRateLimitError(
                    f"Rate limit exceeded for {self.provider_id}", self.provider_id
                ) from e

            # Check for auth error
            if "auth" in error_msg or "401" in error_msg or "403" in error_msg:
                raise ProviderAuthError(
                    f"Authentication failed for {self.provider_id}", self.provider_id
                ) from e

            # Generic error
            raise ProviderError(
                f"Chat completion failed for {self.provider_id}: {str(e)}",
                self.provider_id,
                original_error=e,
                retryable=True,
            ) from e

    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion."""
        try:
            litellm_messages = self._convert_messages(messages)
            full_model = self.get_model_string(model)

            call_kwargs = {
                "model": full_model,
                "messages": litellm_messages,
                "temperature": temperature,
                "stream": True,
                "timeout": self.timeout,
            }

            if max_tokens:
                call_kwargs["max_tokens"] = max_tokens

            if self.base_url and self._provider_type == "ollama":
                call_kwargs["api_base"] = self.base_url

            call_kwargs.update(kwargs)

            response = await litellm.acompletion(**call_kwargs)

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ProviderError(
                f"Streaming chat completion failed for {self.provider_id}: {str(e)}",
                self.provider_id,
                original_error=e,
            ) from e

    async def generate_embedding(
        self, text: str, model: str, **kwargs
    ) -> EmbeddingResponse:
        """Generate an embedding using LiteLLM."""
        try:
            full_model = self.get_model_string(model)

            call_kwargs = {
                "model": full_model,
                "input": text,
                "timeout": self.timeout,
            }

            if self.base_url and self._provider_type == "ollama":
                call_kwargs["api_base"] = self.base_url

            call_kwargs.update(kwargs)

            logger.debug(f"Generating embedding with model: {full_model}")
            response = await litellm.aembedding(**call_kwargs)

            embedding = response.data[0]["embedding"]
            dimension = len(embedding)

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return EmbeddingResponse(
                embedding=embedding,
                model=model,
                provider_id=self.provider_id,
                dimension=dimension,
                usage=usage,
                metadata={"litellm_model": full_model},
            )

        except Exception as e:
            raise ProviderError(
                f"Embedding generation failed for {self.provider_id}: {str(e)}",
                self.provider_id,
                original_error=e,
            ) from e

    async def list_models(self) -> List[ModelInfo]:
        """List available models from configuration.

        Returns models that were configured for this provider.
        For dynamic model discovery, providers should implement
        their own API queries (e.g., Ollama's /api/tags endpoint).

        Returns:
            List of ModelInfo objects from configuration
        """
        models = []

        for model_config in self.config.get("models", []):
            model_info = ModelInfo(
                model_id=model_config["model_id"],
                provider_id=self.provider_id,
                display_name=model_config.get("display_name"),
                context_window=model_config.get("context_window"),
                supports_vision=model_config.get("supports_vision", False),
                supports_audio=model_config.get("supports_audio", False),
                supports_function_calling=model_config.get(
                    "supports_function_calling", False
                ),
                dimension=model_config.get("dimension"),
                metadata={
                    "task": model_config.get("task"),
                    "provider_type": self._provider_type,
                },
            )
            models.append(model_info)

        logger.debug(f"Listing {len(models)} configured models for {self.provider_id}")
        return models

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        start_time = datetime.now()

        try:
            # Try a simple completion with minimal tokens
            test_messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

            # Get first available model from config
            test_model = self.config.get("models", [{}])[0].get("model_id")
            if not test_model:
                return {
                    "status": "unhealthy",
                    "latency_ms": 0,
                    "error": "No models configured",
                    "models_available": 0,
                    "last_check": datetime.now().isoformat(),
                }

            await self.chat_completion(
                messages=test_messages,
                model=test_model,
                max_tokens=1,
                temperature=0,
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "error": None,
                "models_available": len(self.config.get("models", [])),
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "error": str(e),
                "models_available": 0,
                "last_check": datetime.now().isoformat(),
            }

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert ChatMessage objects to LiteLLM format."""
        litellm_messages: List[Dict[str, Any]] = []

        for msg in messages:
            # Handle role - it can be either MessageRole enum or already a string
            role_str = msg.role if isinstance(msg.role, str) else msg.role.value

            message_dict: Dict[str, Any] = {
                "role": role_str,
                "content": msg.content,
            }

            # Handle images for vision models
            if msg.images:
                # LiteLLM supports vision in content array format
                content_parts: List[Dict[str, Any]] = [
                    {"type": "text", "text": msg.content}
                ]

                for image in msg.images:
                    if image.startswith(("http://", "https://")):
                        # Image URL
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": image}}
                        )
                    elif image.startswith("data:"):
                        # Already formatted data URL (e.g., data:image/png;base64,...)
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": image}}
                        )
                    else:
                        # Raw base64 - wrap it
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                            }
                        )

                message_dict["content"] = content_parts

            if msg.name:
                message_dict["name"] = msg.name

            if msg.function_call:
                message_dict["function_call"] = msg.function_call

            litellm_messages.append(message_dict)

        return litellm_messages
