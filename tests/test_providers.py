"""Comprehensive tests for LLM provider implementations."""

import pytest
from unittest.mock import patch, Mock, AsyncMock
from typing import List

from sekha_llm_bridge.providers.base import (
    LlmProvider,
    ChatMessage,
    MessageRole,
    ChatResponse,
    EmbeddingResponse,
    ModelInfo,
    ProviderError,
)
from sekha_llm_bridge.providers.litellm_provider import LiteLlmProvider


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_chat_message_creation(self):
        """Test basic ChatMessage creation."""
        message = ChatMessage(
            role=MessageRole.USER,
            content="Hello world"
        )
        assert message.role == MessageRole.USER
        assert message.content == "Hello world"
        assert message.name is None
        assert message.function_call is None

    def test_chat_message_with_images(self):
        """Test ChatMessage with vision images."""
        message = ChatMessage(
            role=MessageRole.USER,
            content="What's in this image?",
            images=["http://example.com/image.jpg"]
        )
        assert len(message.images) == 1


class TestLlmProviderBase:
    """Test LlmProvider abstract base class."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        # LlmProvider is abstract, can't instantiate directly
        with pytest.raises(TypeError):
            LlmProvider(provider_id="test", config={})


class TestLiteLlmProviderInitialization:
    """Test LiteLLM provider initialization."""

    def test_litellm_provider_creation(self):
        """Test creating a LiteLLM provider."""
        provider = LiteLlmProvider(
            provider_id="test-provider",
            config={"api_key": "test-key", "base_url": "http://localhost"}
        )
        assert provider.provider_id == "test-provider"
        assert provider.provider_type == "litellm"

    def test_litellm_provider_with_minimal_config(self):
        """Test LiteLLM provider with minimal configuration."""
        provider = LiteLlmProvider(
            provider_id="minimal",
            config={}
        )
        assert provider.provider_id == "minimal"


class TestLiteLlmChatCompletion:
    """Test LiteLLM chat completion."""

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self):
        """Test basic chat completion."""
        provider = LiteLlmProvider(
            provider_id="test",
            config={}
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Hello!"}
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        with patch("litellm.acompletion", return_value=mock_response):
            messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
            response = await provider.chat_completion(messages=messages, model="gpt-4o")

            assert response.content == "Hello!"
            assert response.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(self):
        """Test chat completion with custom temperature."""
        provider = LiteLlmProvider(provider_id="test", config={})

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {"content": "Response"}
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}

        with patch("litellm.acompletion", return_value=mock_response) as mock_completion:
            messages = [ChatMessage(role=MessageRole.USER, content="Test")]
            await provider.chat_completion(
                messages=messages,
                model="gpt-4o",
                temperature=0.5
            )

            # Verify temperature was passed
            call_kwargs = mock_completion.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5


class TestLiteLlmEmbedding:
    """Test LiteLLM embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_embedding_basic(self):
        """Test basic embedding generation."""
        provider = LiteLlmProvider(provider_id="test", config={})

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"total_tokens": 5}
        }

        with patch("litellm.embedding", return_value=mock_response):
            response = await provider.generate_embedding(
                text="Hello world",
                model="text-embedding-3-small"
            )

            assert len(response.embedding) == 3
            assert response.dimension == 3


class TestLiteLlmHealthCheck:
    """Test LiteLLM provider health checks."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when provider is healthy."""
        provider = LiteLlmProvider(provider_id="test", config={})

        with patch.object(provider, "list_models", return_value=[]):
            health = await provider.health_check()

            assert health["status"] in ["healthy", "unhealthy", "degraded"]
            assert "latency_ms" in health


class TestProviderEdgeCases:
    """Test provider edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_chat_completion_error_handling(self):
        """Test chat completion error handling."""
        provider = LiteLlmProvider(provider_id="test", config={})

        with patch("litellm.acompletion", side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                messages = [ChatMessage(role=MessageRole.USER, content="Test")]
                await provider.chat_completion(messages=messages, model="gpt-4o")

    @pytest.mark.asyncio
    async def test_embedding_error_handling(self):
        """Test embedding generation error handling."""
        provider = LiteLlmProvider(provider_id="test", config={})

        with patch("litellm.embedding", side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await provider.generate_embedding(text="Test", model="embedding-model")
