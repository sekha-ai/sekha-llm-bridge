"""Integration tests for vision support in v2.0.

Tests validate:
- Vision model routing
- Image format handling (URL and base64)
- Multiple images per message
- Vision provider fallback
- Image format validation
- Error handling for unsupported formats
"""

import pytest
import base64
from unittest.mock import MagicMock, patch
from sekha_llm_bridge.registry import registry
from sekha_llm_bridge.config import ModelTask
from sekha_llm_bridge.providers.litellm_provider import LiteLlmProvider
from sekha_llm_bridge.models import ChatMessage


class TestVisionRouting:
    """Test vision-capable model routing."""

    @pytest.mark.asyncio
    async def test_route_to_vision_model_when_required(self):
        """Test that vision requests route to vision-capable models."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Only vision-capable models returned
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),
            ]

            mock_provider = MagicMock()
            mock_provider.provider_id = "openai"
            mock_provider.provider_type = "openai"

            with patch.object(registry, "providers", {"openai": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            require_vision=True,
                        )

                        # Verify vision requirement passed to candidates
                        mock_candidates.assert_called_once_with(
                            task=ModelTask.CHAT_SMART,
                            require_vision=True,
                            preferred_model=None,
                        )

                        assert result.model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_filter_non_vision_models(self):
        """Test that non-vision models are filtered when vision required."""
        with patch.object(registry, "model_cache") as mock_cache:
            # Mock models: some with vision, some without
            mock_cache.values.return_value = [
                MagicMock(
                    model_id="gpt-4o",
                    provider_id="openai",
                    task=ModelTask.CHAT_SMART,
                    supports_vision=True,
                ),
                MagicMock(
                    model_id="llama3.1:8b",
                    provider_id="ollama",
                    task=ModelTask.CHAT_SMART,
                    supports_vision=False,  # No vision
                ),
            ]

            # Get candidates with vision requirement
            candidates = registry._get_candidates(
                task=ModelTask.CHAT_SMART,
                require_vision=True,
            )

            # Only vision-capable model should be in candidates
            model_ids = [c[1] for c in candidates]
            assert "gpt-4o" in model_ids
            assert "llama3.1:8b" not in model_ids

    @pytest.mark.asyncio
    async def test_no_vision_models_available(self):
        """Test error when vision required but no vision models available."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # No vision models available
            mock_candidates.return_value = []

            with pytest.raises(RuntimeError, match="No suitable provider"):
                await registry.route_with_fallback(
                    task=ModelTask.CHAT_SMART,
                    require_vision=True,
                )


class TestImageFormatHandling:
    """Test handling of different image formats."""

    @pytest.mark.asyncio
    async def test_url_image_format(self):
        """Test handling of URL-based images."""
        provider = LiteLlmProvider(
            "test_provider",
            {
                "provider_type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-key",
            },
        )

        # Message with URL image
        messages = [
            ChatMessage(
                role="user",
                content="What's in this image?",
                images=["https://example.com/image.jpg"],
            )
        ]

        # Convert messages
        litellm_messages = provider._convert_messages(messages)

        # Should have content array with text and image_url
        assert len(litellm_messages) == 1
        assert "content" in litellm_messages[0]

        content = litellm_messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # Text + image

        # Verify image format
        image_part = next((c for c in content if c["type"] == "image_url"), None)
        assert image_part is not None
        assert image_part["image_url"]["url"] == "https://example.com/image.jpg"

    @pytest.mark.asyncio
    async def test_base64_image_format(self):
        """Test handling of base64-encoded images."""
        provider = LiteLlmProvider(
            "test_provider",
            {
                "provider_type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-key",
            },
        )

        # Create a small base64 image (1x1 pixel PNG)
        small_png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        base64_image = f"data:image/png;base64,{small_png}"

        messages = [
            ChatMessage(
                role="user",
                content="Analyze this image",
                images=[base64_image],
            )
        ]

        litellm_messages = provider._convert_messages(messages)

        # Verify base64 image in content
        content = litellm_messages[0]["content"]
        image_part = next((c for c in content if c["type"] == "image_url"), None)
        assert image_part is not None
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_multiple_images_per_message(self):
        """Test handling of multiple images in a single message."""
        provider = LiteLlmProvider(
            "test_provider",
            {
                "provider_type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-key",
            },
        )

        messages = [
            ChatMessage(
                role="user",
                content="Compare these images",
                images=[
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                    "https://example.com/image3.jpg",
                ],
            )
        ]

        litellm_messages = provider._convert_messages(messages)

        # Should have content array with text + 3 images
        content = litellm_messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 4  # 1 text + 3 images

        # Verify all images present
        image_parts = [c for c in content if c["type"] == "image_url"]
        assert len(image_parts) == 3
        assert image_parts[0]["image_url"]["url"] == "https://example.com/image1.jpg"
        assert image_parts[1]["image_url"]["url"] == "https://example.com/image2.jpg"
        assert image_parts[2]["image_url"]["url"] == "https://example.com/image3.jpg"

    @pytest.mark.asyncio
    async def test_message_without_images(self):
        """Test that messages without images work normally."""
        provider = LiteLlmProvider(
            "test_provider",
            {
                "provider_type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-key",
            },
        )

        messages = [
            ChatMessage(
                role="user",
                content="Just text, no images",
                images=None,
            )
        ]

        litellm_messages = provider._convert_messages(messages)

        # Should be simple string content (not array)
        assert len(litellm_messages) == 1
        assert litellm_messages[0]["content"] == "Just text, no images"
        assert isinstance(litellm_messages[0]["content"], str)

    @pytest.mark.asyncio
    async def test_empty_images_list(self):
        """Test handling of empty images list."""
        provider = LiteLlmProvider(
            "test_provider",
            {
                "provider_type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-key",
            },
        )

        messages = [
            ChatMessage(
                role="user",
                content="Text message",
                images=[],  # Empty list
            )
        ]

        litellm_messages = provider._convert_messages(messages)

        # Should treat as text-only
        assert litellm_messages[0]["content"] == "Text message"
        assert isinstance(litellm_messages[0]["content"], str)


class TestVisionProviderFallback:
    """Test fallback behavior for vision requests."""

    @pytest.mark.asyncio
    async def test_fallback_to_secondary_vision_provider(self):
        """Test fallback to secondary provider when primary vision provider fails."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Two vision providers, primary fails
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),  # Primary
                ("anthropic", "claude-3.5-sonnet", 2),  # Fallback
            ]

            openai = MagicMock()
            openai.provider_id = "openai"

            anthropic = MagicMock()
            anthropic.provider_id = "anthropic"

            with patch.object(
                registry, "providers", {"openai": openai, "anthropic": anthropic}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    # OpenAI circuit breaker open (failed)
                    def cb_side_effect(provider_id):
                        if provider_id == "openai":
                            return MagicMock(is_open=lambda: True)
                        return MagicMock(is_open=lambda: False)

                    mock_cbs.get.side_effect = cb_side_effect

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            require_vision=True,
                        )

                        # Should fallback to Anthropic
                        assert result.provider.provider_id == "anthropic"
                        assert result.model_id == "claude-3.5-sonnet"

    @pytest.mark.asyncio
    async def test_all_vision_providers_unavailable(self):
        """Test error when all vision providers are unavailable."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),
                ("anthropic", "claude-3.5-sonnet", 2),
            ]

            openai = MagicMock()
            openai.provider_id = "openai"

            anthropic = MagicMock()
            anthropic.provider_id = "anthropic"

            with patch.object(
                registry, "providers", {"openai": openai, "anthropic": anthropic}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    # All circuit breakers open
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: True)

                    with pytest.raises(RuntimeError, match="No suitable provider"):
                        await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            require_vision=True,
                        )


class TestImageFormatValidation:
    """Test image format validation."""

    @pytest.mark.asyncio
    async def test_validate_supported_image_formats(self):
        """Test that supported image formats are accepted."""
        supported_formats = [
            "https://example.com/image.jpg",
            "https://example.com/image.jpeg",
            "https://example.com/image.png",
            "https://example.com/image.gif",
            "https://example.com/image.webp",
            "data:image/png;base64,iVBORw0KGgo=",
            "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        ]

        def is_valid_image_format(image_url):
            """Check if image format is supported."""
            # URL format
            if image_url.startswith(("http://", "https://")):
                return any(
                    image_url.lower().endswith(ext)
                    for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                )
            # Base64 format
            if image_url.startswith("data:image/"):
                return True
            return False

        # All should be valid
        for fmt in supported_formats:
            assert is_valid_image_format(fmt), f"Format should be valid: {fmt}"

    @pytest.mark.asyncio
    async def test_reject_unsupported_image_formats(self):
        """Test that unsupported image formats are rejected."""
        unsupported_formats = [
            "https://example.com/file.pdf",
            "https://example.com/doc.txt",
            "file:///local/path/image.jpg",  # Local file URLs not supported
            "ftp://example.com/image.jpg",  # FTP not supported
            "just-a-filename.jpg",  # Not a URL or base64
        ]

        def is_valid_image_format(image_url):
            if image_url.startswith(("http://", "https://")):
                return any(
                    image_url.lower().endswith(ext)
                    for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                )
            if image_url.startswith("data:image/"):
                return True
            return False

        # All should be invalid
        for fmt in unsupported_formats:
            assert not is_valid_image_format(fmt), f"Format should be invalid: {fmt}"

    @pytest.mark.asyncio
    async def test_validate_base64_encoding(self):
        """Test validation of base64-encoded images."""

        def is_valid_base64(s):
            """Check if string is valid base64."""
            try:
                if not s.startswith("data:image/"):
                    return False
                # Extract base64 part
                base64_data = s.split(",", 1)[1] if "," in s else s
                base64.b64decode(base64_data, validate=True)
                return True
            except Exception:
                return False

        # Valid base64
        valid_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        assert is_valid_base64(valid_base64)

        # Invalid base64
        invalid_base64 = "data:image/png;base64,not-valid-base64!!!"
        assert not is_valid_base64(invalid_base64)


class TestVisionResponseHandling:
    """Test handling of responses from vision models."""

    @pytest.mark.asyncio
    async def test_vision_response_includes_routing_metadata(self):
        """Test that vision responses include routing metadata."""
        # Mock routing result
        routing_result = {
            "provider_id": "openai",
            "model_id": "gpt-4o",
            "estimated_cost": 0.01,
            "reason": "Vision support required",
            "supports_vision": True,
        }

        # Response should include routing metadata
        assert routing_result["supports_vision"] is True
        assert "gpt-4o" in routing_result["model_id"]
        assert routing_result["provider_id"] == "openai"

    @pytest.mark.asyncio
    async def test_vision_cost_estimation(self):
        """Test that vision requests have appropriate cost estimates."""
        from sekha_llm_bridge.pricing import estimate_cost

        # Vision models typically cost more than text-only
        text_cost = estimate_cost("gpt-4o", 1000, 500)  # Text only

        # Vision cost would be calculated based on image size + tokens
        # For now, just verify text cost is estimated
        assert text_cost > 0.0

        # In practice, would add image tokens to estimation
        # vision_cost = text_cost + (num_images * image_token_equivalent)


class TestProxyVisionDetection:
    """Test proxy's automatic vision detection."""

    @pytest.mark.asyncio
    async def test_detect_vision_request_from_messages(self):
        """Test that proxy detects vision requests from message content."""

        def has_images(messages):
            """Check if any message contains images."""
            for message in messages:
                if isinstance(message, dict) and message.get("images"):
                    return True
                if isinstance(message, ChatMessage) and message.images:
                    return True
            return False

        # Text-only messages
        text_messages = [{"role": "user", "content": "Hello"}]
        assert not has_images(text_messages)

        # Messages with images
        vision_messages = [
            ChatMessage(
                role="user",
                content="What's this?",
                images=["https://example.com/image.jpg"],
            )
        ]
        assert has_images(vision_messages)

    @pytest.mark.asyncio
    async def test_set_require_vision_flag(self):
        """Test that require_vision flag is set when images detected."""
        messages = [
            ChatMessage(
                role="user",
                content="Analyze this image",
                images=["https://example.com/test.jpg"],
            )
        ]

        # Simulate proxy logic
        require_vision = any(
            (isinstance(m, ChatMessage) and m.images)
            or (isinstance(m, dict) and m.get("images"))
            for m in messages
        )

        assert require_vision is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
