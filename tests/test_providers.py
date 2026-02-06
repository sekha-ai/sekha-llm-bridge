"""Comprehensive tests for provider implementations."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from sekha_llm_bridge.providers.base import LlmProvider, ProviderCapabilities
from sekha_llm_bridge.providers.litellm_provider import LiteLlmProvider
from sekha_llm_bridge.models.requests import (
    ChatCompletionRequest,
    EmbeddingRequest,
    Message,
)


class TestProviderCapabilities:
    """Test ProviderCapabilities dataclass."""

    def test_capabilities_defaults(self):
        """Test default capability values."""
        caps = ProviderCapabilities()

        assert not caps.supports_streaming
        assert not caps.supports_function_calling
        assert not caps.supports_vision
        assert not caps.supports_audio
        assert caps.max_context_tokens == 4096
        assert caps.supported_models == []

    def test_capabilities_custom_values(self):
        """Test custom capability values."""
        caps = ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            supports_audio=False,
            max_context_tokens=128000,
            supported_models=["gpt-4o", "gpt-4o-mini"],
        )

        assert caps.supports_streaming
        assert caps.supports_function_calling
        assert caps.supports_vision
        assert not caps.supports_audio
        assert caps.max_context_tokens == 128000
        assert len(caps.supported_models) == 2


class TestLlmProviderBase:
    """Test LlmProvider base class."""

    def test_provider_cannot_be_instantiated(self):
        """Test LlmProvider base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LlmProvider("test", {})

    def test_provider_subclass_must_implement_methods(self):
        """Test subclass must implement abstract methods."""

        class IncompleteProvider(LlmProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider("test", {})

    def test_provider_subclass_with_all_methods(self):
        """Test complete provider subclass can be instantiated."""

        class CompleteProvider(LlmProvider):
            async def chat_completion(self, request, **kwargs):
                pass

            async def embedding(self, request, **kwargs):
                pass

            def get_capabilities(self):
                return ProviderCapabilities()

            async def health_check(self):
                return True

        provider = CompleteProvider("test", {})
        assert provider.provider_id == "test"


class TestLiteLlmProviderInitialization:
    """Test LiteLlmProvider initialization."""

    def test_provider_initialization_minimal(self):
        """Test provider initialization with minimal config."""
        config = {"provider_type": "litellm", "models": []}

        provider = LiteLlmProvider("test-provider", config)

        assert provider.provider_id == "test-provider"
        assert provider.provider_type == "litellm"

    def test_provider_initialization_with_base_url(self):
        """Test provider initialization with base URL."""
        config = {
            "provider_type": "litellm",
            "base_url": "http://localhost:11434",
            "models": [],
        }

        provider = LiteLlmProvider("ollama", config)

        assert provider.base_url == "http://localhost:11434"

    def test_provider_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        config = {
            "provider_type": "litellm",
            "api_key": "sk-test-key",
            "models": [],
        }

        provider = LiteLlmProvider("openai", config)

        assert provider.api_key == "sk-test-key"

    def test_provider_initialization_with_timeout(self):
        """Test provider initialization with custom timeout."""
        config = {"provider_type": "litellm", "timeout": 60, "models": []}

        provider = LiteLlmProvider("test", config)

        assert provider.timeout == 60

    def test_provider_initialization_with_models(self):
        """Test provider initialization with model list."""
        config = {
            "provider_type": "litellm",
            "models": [
                {"model_id": "gpt-4o", "task": "chat_smart", "context_window": 128000},
                {
                    "model_id": "gpt-4o-mini",
                    "task": "chat_small",
                    "context_window": 128000,
                },
            ],
        }

        provider = LiteLlmProvider("openai", config)

        assert len(provider.models) == 2
        assert provider.models[0]["model_id"] == "gpt-4o"

    def test_provider_initialization_defaults(self):
        """Test provider initialization uses defaults when not specified."""
        config = {"provider_type": "litellm"}

        provider = LiteLlmProvider("test", config)

        assert provider.base_url is None
        assert provider.api_key is None
        assert provider.timeout == 30  # Default timeout
        assert provider.models == []


class TestLiteLlmChatCompletion:
    """Test LiteLlmProvider chat_completion method."""

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self):
        """Test basic chat completion."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = ChatCompletionRequest(
            model="gpt-4o", messages=[Message(role="user", content="Hello")]
        )

        mock_response = {
            "id": "test-id",
            "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            response = await provider.chat_completion(request)

            assert response["id"] == "test-id"
            assert response["choices"][0]["message"]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_chat_completion_with_streaming(self):
        """Test chat completion with streaming."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hello")],
            stream=True,
        )

        async def mock_stream():
            yield {"choices": [{"delta": {"content": "Hi"}}]}
            yield {"choices": [{"delta": {"content": " there!"}}]}

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_stream())):
            response = await provider.chat_completion(request)

            # Should return async generator
            assert hasattr(response, "__aiter__")

    @pytest.mark.asyncio
    async def test_chat_completion_with_functions(self):
        """Test chat completion with function calling."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="What's the weather?")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        mock_response = {
            "id": "test-id",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "get_weather"}}],
                    }
                }
            ],
        }

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            response = await provider.chat_completion(request)

            assert "tool_calls" in response["choices"][0]["message"]

    @pytest.mark.asyncio
    async def test_chat_completion_with_system_message(self):
        """Test chat completion with system message."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                Message(role="system", content="You are helpful"),
                Message(role="user", content="Hello"),
            ],
        )

        mock_response = {
            "id": "test-id",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
        }

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            response = await provider.chat_completion(request)

            assert response is not None

    @pytest.mark.asyncio
    async def test_chat_completion_error_handling(self):
        """Test chat completion error handling."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = ChatCompletionRequest(
            model="gpt-4o", messages=[Message(role="user", content="Hello")]
        )

        with patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=Exception("API Error")),
        ):
            with pytest.raises(Exception, match="API Error"):
                await provider.chat_completion(request)

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(self):
        """Test chat completion with temperature setting."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
        )

        mock_response = {
            "id": "test-id",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
        }

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            response = await provider.chat_completion(request)

            assert response is not None


class TestLiteLlmEmbedding:
    """Test LiteLlmProvider embedding method."""

    @pytest.mark.asyncio
    async def test_embedding_single_input(self):
        """Test embedding with single input."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = EmbeddingRequest(model="text-embedding-3-small", input="Hello world")

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"total_tokens": 2},
        }

        with patch("litellm.aembedding", new=AsyncMock(return_value=mock_response)):
            response = await provider.embedding(request)

            assert len(response["data"]) == 1
            assert len(response["data"][0]["embedding"]) == 3

    @pytest.mark.asyncio
    async def test_embedding_multiple_inputs(self):
        """Test embedding with multiple inputs."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = EmbeddingRequest(
            model="text-embedding-3-small", input=["Hello", "World"]
        )

        mock_response = {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ],
            "usage": {"total_tokens": 2},
        }

        with patch("litellm.aembedding", new=AsyncMock(return_value=mock_response)):
            response = await provider.embedding(request)

            assert len(response["data"]) == 2

    @pytest.mark.asyncio
    async def test_embedding_error_handling(self):
        """Test embedding error handling."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = EmbeddingRequest(model="text-embedding-3-small", input="Hello")

        with patch(
            "litellm.aembedding", new=AsyncMock(side_effect=Exception("API Error"))
        ):
            with pytest.raises(Exception, match="API Error"):
                await provider.embedding(request)

    @pytest.mark.asyncio
    async def test_embedding_with_dimensions(self):
        """Test embedding with custom dimensions."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        request = EmbeddingRequest(
            model="text-embedding-3-small", input="Hello", dimensions=256
        )

        mock_response = {
            "data": [{"embedding": [0.1] * 256, "index": 0}],
            "usage": {"total_tokens": 1},
        }

        with patch("litellm.aembedding", new=AsyncMock(return_value=mock_response)):
            response = await provider.embedding(request)

            assert len(response["data"][0]["embedding"]) == 256


class TestLiteLlmCapabilities:
    """Test LiteLlmProvider get_capabilities method."""

    def test_get_capabilities(self):
        """Test getting provider capabilities."""
        config = {
            "provider_type": "litellm",
            "models": [
                {"model_id": "gpt-4o", "task": "chat_smart", "context_window": 128000}
            ],
        }
        provider = LiteLlmProvider("openai", config)

        capabilities = provider.get_capabilities()

        assert isinstance(capabilities, ProviderCapabilities)
        assert capabilities.supports_streaming
        assert capabilities.supports_function_calling

    def test_capabilities_reflect_models(self):
        """Test capabilities reflect configured models."""
        config = {
            "provider_type": "litellm",
            "models": [
                {"model_id": "gpt-4o", "task": "chat_smart", "context_window": 128000},
                {
                    "model_id": "gpt-4o-mini",
                    "task": "chat_small",
                    "context_window": 128000,
                },
            ],
        }
        provider = LiteLlmProvider("openai", config)

        capabilities = provider.get_capabilities()

        assert len(capabilities.supported_models) == 2


class TestLiteLlmHealthCheck:
    """Test LiteLlmProvider health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        # Mock a simple completion call
        mock_response = {
            "id": "test",
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
        }

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            is_healthy = await provider.health_check()

            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        config = {"provider_type": "litellm", "models": []}
        provider = LiteLlmProvider("test", config)

        with patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=Exception("Connection failed")),
        ):
            is_healthy = await provider.health_check()

            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_with_base_url(self):
        """Test health check uses configured base URL."""
        config = {
            "provider_type": "litellm",
            "base_url": "http://localhost:11434",
            "models": [],
        }
        provider = LiteLlmProvider("ollama", config)

        mock_response = {"id": "test", "choices": [{"message": {"content": "OK"}}]}

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            is_healthy = await provider.health_check()

            assert is_healthy is True


class TestProviderEdgeCases:
    """Test edge cases and error scenarios."""

    def test_provider_with_empty_config(self):
        """Test provider with minimal empty config."""
        provider = LiteLlmProvider("test", {})

        assert provider.provider_id == "test"
        assert provider.models == []

    def test_provider_id_validation(self):
        """Test provider ID is set correctly."""
        provider = LiteLlmProvider("my-custom-provider-123", {})

        assert provider.provider_id == "my-custom-provider-123"

    @pytest.mark.asyncio
    async def test_chat_with_empty_messages(self):
        """Test chat completion with empty messages list."""
        config = {"provider_type": "litellm"}
        provider = LiteLlmProvider("test", config)

        request = ChatCompletionRequest(model="gpt-4o", messages=[])

        # Should still call litellm even with empty messages
        mock_response = {"id": "test", "choices": []}

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            response = await provider.chat_completion(request)

            assert response["id"] == "test"

    def test_capabilities_without_models(self):
        """Test getting capabilities when no models configured."""
        provider = LiteLlmProvider("test", {"models": []})

        capabilities = provider.get_capabilities()

        assert capabilities.supported_models == []
        assert capabilities.max_context_tokens == 4096  # Default
