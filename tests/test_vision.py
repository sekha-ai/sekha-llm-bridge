"""Tests for vision support and multi-modal messages."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from sekha_llm_bridge.providers.base import (
    ChatMessage,
    MessageRole,
    ChatResponse,
    ModelInfo,
)
from sekha_llm_bridge.providers.litellm_provider import LiteLlmProvider
from sekha_llm_bridge.registry import ModelRegistry, RoutingResult
from sekha_llm_bridge.config import ModelTask


class TestVisionMessageConversion:
    """Test conversion of vision messages to provider format."""

    def test_text_only_message(self):
        """Test simple text message conversion."""
        provider = LiteLlmProvider(
            "test_provider",
            {"provider_type": "openai", "api_key": "test"},
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        converted = provider._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"

    def test_message_with_image_url(self):
        """Test message with image URL conversion."""
        provider = LiteLlmProvider(
            "test_provider",
            {"provider_type": "openai", "api_key": "test"},
        )

        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="What's in this image?",
                images=["https://example.com/photo.jpg"],
            )
        ]
        converted = provider._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        assert len(converted[0]["content"]) == 2

        # Check text part
        text_part = converted[0]["content"][0]
        assert text_part["type"] == "text"
        assert text_part["text"] == "What's in this image?"

        # Check image part
        image_part = converted[0]["content"][1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"] == "https://example.com/photo.jpg"

    def test_message_with_base64_image(self):
        """Test message with base64 image conversion."""
        provider = LiteLlmProvider(
            "test_provider",
            {"provider_type": "openai", "api_key": "test"},
        )

        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="Describe this image",
                images=[base64_data],
            )
        ]
        converted = provider._convert_messages(messages)

        assert len(converted) == 1
        image_part = converted[0]["content"][1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
        assert base64_data in image_part["image_url"]["url"]

    def test_message_with_multiple_images(self):
        """Test message with multiple images."""
        provider = LiteLlmProvider(
            "test_provider",
            {"provider_type": "openai", "api_key": "test"},
        )

        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="Compare these images",
                images=[
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                    "base64encodedimage123",
                ],
            )
        ]
        converted = provider._convert_messages(messages)

        assert len(converted) == 1
        content = converted[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 4  # 1 text + 3 images

        # Check text
        assert content[0]["type"] == "text"

        # Check images
        assert content[1]["type"] == "image_url"
        assert "image1.jpg" in content[1]["image_url"]["url"]

        assert content[2]["type"] == "image_url"
        assert "image2.jpg" in content[2]["image_url"]["url"]

        assert content[3]["type"] == "image_url"
        assert "base64" in content[3]["image_url"]["url"]

    def test_multi_turn_with_vision(self):
        """Test multi-turn conversation with vision."""
        provider = LiteLlmProvider(
            "test_provider",
            {"provider_type": "openai", "api_key": "test"},
        )

        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="What's in this image?",
                images=["https://example.com/photo.jpg"],
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I see a cat in the image.",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="What color is it?",
            ),
        ]
        converted = provider._convert_messages(messages)

        assert len(converted) == 3

        # First message has image
        assert isinstance(converted[0]["content"], list)
        assert len(converted[0]["content"]) == 2

        # Second message is text only
        assert converted[1]["content"] == "I see a cat in the image."

        # Third message is text only
        assert converted[2]["content"] == "What color is it?"


class TestVisionRouting:
    """Test routing for vision requests."""

    @pytest.mark.asyncio
    async def test_require_vision_filters_models(self):
        """Test that require_vision filters to vision-capable models."""
        # Mock settings with vision and non-vision models
        with patch("sekha_llm_bridge.registry.settings") as mock_settings:
            mock_settings.providers = [
                Mock(
                    id="provider1",
                    provider_type="openai",
                    base_url="http://test",
                    api_key="test",
                    timeout_secs=30,
                    priority=1,
                    models=[
                        Mock(
                            model_id="gpt-4o",
                            task=ModelTask.CHAT_SMART,
                            context_window=128000,
                            dimension=None,
                            supports_vision=True,
                            supports_audio=False,
                        ),
                        Mock(
                            model_id="gpt-3.5-turbo",
                            task=ModelTask.CHAT_SMALL,
                            context_window=16000,
                            dimension=None,
                            supports_vision=False,
                            supports_audio=False,
                        ),
                    ],
                )
            ]
            mock_settings.routing = Mock(
                cost_aware=True,
                max_cost_per_request=1.0,
                circuit_breaker=Mock(
                    failure_threshold=5,
                    timeout_secs=60,
                    success_threshold=2,
                ),
            )

            registry = ModelRegistry()

            # Request with vision requirement
            result = await registry.route_with_fallback(
                task=ModelTask.CHAT_SMART,
                require_vision=True,
            )

            assert result.model_id == "gpt-4o"
            assert "gpt-3.5-turbo" not in result.model_id

    @pytest.mark.asyncio
    async def test_no_vision_models_raises_error(self):
        """Test error when no vision models available."""
        with patch("sekha_llm_bridge.registry.settings") as mock_settings:
            mock_settings.providers = [
                Mock(
                    id="provider1",
                    provider_type="openai",
                    base_url="http://test",
                    api_key="test",
                    timeout_secs=30,
                    priority=1,
                    models=[
                        Mock(
                            model_id="gpt-3.5-turbo",
                            task=ModelTask.CHAT_SMALL,
                            context_window=16000,
                            dimension=None,
                            supports_vision=False,
                            supports_audio=False,
                        ),
                    ],
                )
            ]
            mock_settings.routing = Mock(
                cost_aware=True,
                max_cost_per_request=1.0,
                circuit_breaker=Mock(
                    failure_threshold=5,
                    timeout_secs=60,
                    success_threshold=2,
                ),
            )

            registry = ModelRegistry()

            with pytest.raises(RuntimeError, match="No providers available"):
                await registry.route_with_fallback(
                    task=ModelTask.CHAT_SMALL,
                    require_vision=True,
                )

    @pytest.mark.asyncio
    async def test_vision_routing_respects_cost_limit(self):
        """Test that vision routing respects cost limits."""
        with patch("sekha_llm_bridge.registry.settings") as mock_settings:
            with patch("sekha_llm_bridge.registry.estimate_cost") as mock_cost:
                # Expensive model
                mock_cost.side_effect = lambda model, *args, **kwargs: (
                    0.10 if "gpt-4o" in model else 0.001
                )

                mock_settings.providers = [
                    Mock(
                        id="provider1",
                        provider_type="openai",
                        base_url="http://test",
                        api_key="test",
                        timeout_secs=30,
                        priority=1,
                        models=[
                            Mock(
                                model_id="gpt-4o",
                                task=ModelTask.CHAT_SMART,
                                context_window=128000,
                                dimension=None,
                                supports_vision=True,
                                supports_audio=False,
                            ),
                            Mock(
                                model_id="gpt-4o-mini",
                                task=ModelTask.CHAT_SMART,
                                context_window=128000,
                                dimension=None,
                                supports_vision=True,
                                supports_audio=False,
                            ),
                        ],
                    )
                ]
                mock_settings.routing = Mock(
                    cost_aware=True,
                    max_cost_per_request=1.0,
                    circuit_breaker=Mock(
                        failure_threshold=5,
                        timeout_secs=60,
                        success_threshold=2,
                    ),
                )

                registry = ModelRegistry()

                # Request with low cost limit
                result = await registry.route_with_fallback(
                    task=ModelTask.CHAT_SMART,
                    require_vision=True,
                    max_cost=0.01,
                )

                # Should pick cheaper model
                assert result.model_id == "gpt-4o-mini"
                assert result.estimated_cost <= 0.01


class TestVisionIntegration:
    """Integration tests for end-to-end vision support."""

    @pytest.mark.asyncio
    async def test_vision_request_flow(self):
        """Test complete flow from message to provider."""
        provider = LiteLlmProvider(
            "test_provider",
            {
                "provider_type": "openai",
                "api_key": "test",
                "models": [{"model_id": "gpt-4o", "task": "chat_smart"}],
            },
        )

        # Create vision message
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="What's in this image?",
                images=["https://example.com/photo.jpg"],
            )
        ]

        # Mock LiteLLM response
        mock_response = Mock(
            choices=[
                Mock(
                    message=Mock(content="I see a beautiful sunset"),
                    finish_reason="stop",
                )
            ],
            usage=Mock(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

        with patch("litellm.acompletion", return_value=mock_response):
            response = await provider.chat_completion(
                messages=messages,
                model="gpt-4o",
                temperature=0.7,
            )

        assert response.content == "I see a beautiful sunset"
        assert response.model == "gpt-4o"
        assert response.usage["total_tokens"] == 120

    def test_image_url_detection(self):
        """Test detection of image URLs vs base64."""
        provider = LiteLlmProvider(
            "test_provider",
            {"provider_type": "openai", "api_key": "test"},
        )

        # HTTP URL
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="Test",
                images=["https://example.com/photo.jpg"],
            )
        ]
        converted = provider._convert_messages(messages)
        assert (
            converted[0]["content"][1]["image_url"]["url"]
            == "https://example.com/photo.jpg"
        )

        # Base64
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="Test",
                images=["base64data123"],
            )
        ]
        converted = provider._convert_messages(messages)
        assert converted[0]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
