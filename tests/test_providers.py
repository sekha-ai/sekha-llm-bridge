"""Comprehensive tests for LLM provider implementations."""

from unittest.mock import Mock, patch

import pytest

from sekha_llm_bridge.providers.base import ChatMessage, LlmProvider, MessageRole
from sekha_llm_bridge.providers.litellm_provider import LiteLlmProvider


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_chat_message_creation(self):
        """Test basic ChatMessage creation."""
        message = ChatMessage(role=MessageRole.USER, content="Hello world")
        assert message.role == MessageRole.USER
        assert message.content == "Hello world"
        assert message.name is None
        assert message.function_call is None

    def test_chat_message_with_images(self):
        """Test ChatMessage with vision images."""
        message = ChatMessage(
            role=MessageRole.USER,
            content="What's in this image?",
            images=["http://example.com/image.jpg"],
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
            config={
                "provider_type": "ollama",
                "api_key": "test-key",
                "base_url": "http://localhost",
            },
        )
        assert provider.provider_id == "test-provider"
        # Provider type comes from config, defaults to "ollama"
        assert provider.provider_type == "ollama"

    def test_litellm_provider_with_minimal_config(self):
        """Test LiteLLM provider with minimal configuration."""
        provider = LiteLlmProvider(provider_id="minimal", config={})
        assert provider.provider_id == "minimal"
        # Defaults to ollama when no provider_type specified
        assert provider.provider_type == "ollama"


class TestLiteLlmChatCompletion:
    """Test LiteLLM chat completion."""

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self):
        """Test basic chat completion."""
        provider = LiteLlmProvider(provider_id="test", config={})

        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Mock message as object with .content attribute
        mock_message = Mock()
        mock_message.content = "Hello!"
        mock_response.choices[0].message = mock_message
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

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
        mock_message = Mock()
        mock_message.content = "Response"
        mock_response.choices[0].message = mock_message
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_usage = Mock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 10
        mock_response.usage = mock_usage

        with patch(
            "litellm.acompletion", return_value=mock_response
        ) as mock_completion:
            messages = [ChatMessage(role=MessageRole.USER, content="Test")]
            await provider.chat_completion(
                messages=messages, model="gpt-4o", temperature=0.5
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

        mock_response = Mock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 5
        mock_usage.total_tokens = 5
        mock_response.usage = mock_usage

        with patch("litellm.aembedding", return_value=mock_response):
            response = await provider.generate_embedding(
                text="Hello world", model="text-embedding-3-small"
            )

            assert len(response.embedding) == 3
            assert response.dimension == 3


class TestLiteLlmHealthCheck:
    """Test LiteLLM provider health checks."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when provider is healthy."""
        provider = LiteLlmProvider(
            provider_id="test", config={"models": [{"model_id": "test-model"}]}
        )

        # Mock successful chat completion for health check
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_message = Mock()
        mock_message.content = "Hi"
        mock_response.choices[0].message = mock_message
        mock_response.choices[0].finish_reason = "stop"
        mock_usage = Mock()
        mock_usage.prompt_tokens = 1
        mock_usage.completion_tokens = 1
        mock_usage.total_tokens = 2
        mock_response.usage = mock_usage

        with patch("litellm.acompletion", return_value=mock_response):
            health = await provider.health_check()

            assert health["status"] == "healthy"
            assert "latency_ms" in health
            assert health["models_available"] == 1


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

        with patch("litellm.aembedding", side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await provider.generate_embedding(text="Test", model="embedding-model")
