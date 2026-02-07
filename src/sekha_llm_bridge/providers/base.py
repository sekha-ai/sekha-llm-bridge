"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class MessageRole(str, Enum):
    """Message roles in chat conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    images: Optional[List[str]] = None  # URLs or base64


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    content: str
    model: str
    provider_id: str
    finish_reason: str
    usage: Dict[
        str, int
    ]  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    """Response from an embedding generation."""

    embedding: List[float]
    model: str
    provider_id: str
    dimension: int
    usage: Dict[str, int]  # {"prompt_tokens": X, "total_tokens": X}
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelInfo:
    """Information about an available model."""

    model_id: str
    provider_id: str
    display_name: Optional[str] = None
    context_window: Optional[int] = None
    supports_vision: bool = False
    supports_audio: bool = False
    supports_function_calling: bool = False
    dimension: Optional[int] = None  # For embedding models
    metadata: Optional[Dict[str, Any]] = None


class LlmProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        """Initialize the provider.

        Args:
            provider_id: Unique identifier for this provider instance
            config: Provider-specific configuration
        """
        self.provider_id = provider_id
        self.config = config

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Get the provider type (e.g., 'ollama', 'openai')."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Generate a chat completion.

        Args:
            messages: List of chat messages
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatResponse with the completion

        Raises:
            ProviderError: If the request fails
        """
        pass

    @abstractmethod
    def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion.

        Args:
            messages: List of chat messages
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of the completion text

        Raises:
            ProviderError: If the request fails
        """
        pass

    @abstractmethod
    async def generate_embedding(
        self, text: str, model: str, **kwargs
    ) -> EmbeddingResponse:
        """Generate an embedding for text.

        Args:
            text: Text to embed
            model: Model identifier
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbeddingResponse with the embedding vector

        Raises:
            ProviderError: If the request fails
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models from this provider.

        Returns:
            List of ModelInfo objects

        Raises:
            ProviderError: If the request fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health.

        Returns:
            Dictionary with health status:
            {
                "status": "healthy" | "unhealthy" | "degraded",
                "latency_ms": float,
                "error": Optional[str],
                "models_available": int,
                "last_check": str (ISO timestamp)
            }
        """
        pass

    def get_model_string(self, model: str) -> str:
        """Convert model ID to provider-specific format.

        For example, for LiteLLM:
        - 'llama3.1:8b' -> 'ollama/llama3.1:8b'
        - 'gpt-4o' -> 'openai/gpt-4o'

        Args:
            model: Model identifier

        Returns:
            Provider-specific model string
        """
        return model


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider_id: str,
        original_error: Optional[Exception] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider_id = provider_id
        self.original_error = original_error
        self.retryable = retryable


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""

    def __init__(self, message: str, provider_id: str):
        super().__init__(message, provider_id, retryable=True)


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded."""

    def __init__(
        self, message: str, provider_id: str, retry_after: Optional[int] = None
    ):
        super().__init__(message, provider_id, retryable=True)
        self.retry_after = retry_after


class ProviderAuthError(ProviderError):
    """Provider authentication failed."""

    def __init__(self, message: str, provider_id: str):
        super().__init__(message, provider_id, retryable=False)
