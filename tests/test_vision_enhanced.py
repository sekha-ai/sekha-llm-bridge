"""Tests for enhanced vision detection features.

Tests validate:
- URL pattern detection in text
- Base64 data URI detection
- Image count tracking
- Vision capabilities metadata
"""

import pytest

from sekha_llm_bridge.config import ModelConfig, ModelTask, VisionCapabilities


class TestVisionCapabilities:
    """Test vision capabilities configuration."""

    def test_default_vision_capabilities(self):
        """Test default vision capabilities are set."""
        model = ModelConfig(
            model_id="gpt-4o",
            task=ModelTask.VISION,
            supports_vision=True,
        )

        # Should auto-create default capabilities
        assert model.vision_capabilities is not None
        assert model.vision_capabilities.max_images == 10
        assert model.vision_capabilities.max_image_size_mb == 20
        assert "jpg" in model.vision_capabilities.supported_formats
        assert "png" in model.vision_capabilities.supported_formats

    def test_custom_vision_capabilities(self):
        """Test custom vision capabilities."""
        capabilities = VisionCapabilities(
            max_images=5, max_image_size_mb=10, supported_formats=["jpg", "png"]
        )

        model = ModelConfig(
            model_id="claude-3-opus",
            task=ModelTask.VISION,
            supports_vision=True,
            vision_capabilities=capabilities,
        )

        assert model.vision_capabilities.max_images == 5
        assert model.vision_capabilities.max_image_size_mb == 10
        assert len(model.vision_capabilities.supported_formats) == 2

    def test_no_vision_capabilities_without_support(self):
        """Test models without vision support don't get capabilities."""
        model = ModelConfig(
            model_id="llama3.1:8b",
            task=ModelTask.CHAT_SMALL,
            supports_vision=False,
        )

        assert model.vision_capabilities is None


class TestEnhancedImageDetection:
    """Test enhanced image detection methods."""

    def test_detect_image_url_in_text(self):
        """Test detection of image URLs in text content."""
        import re

        # Pattern from proxy.py
        pattern = re.compile(
            r'https?://[^\s<>"]+\.(?:jpg|jpeg|png|gif|bmp|webp|svg)(?:[?#][^\s<>"]*)?',
            re.IGNORECASE,
        )

        # Test various image URL formats
        test_cases = [
            ("Check this image: https://example.com/photo.jpg", 1),
            ("Here: https://cdn.example.com/images/pic.PNG", 1),
            ("https://site.com/img.jpg and https://other.com/photo.png", 2),
            ("Image with params: https://example.com/pic.jpg?size=large", 1),
            ("With fragment: https://example.com/pic.png#section", 1),
            ("WebP format: https://example.com/image.webp", 1),
            ("No images here", 0),
            ("Just a link: https://example.com/page", 0),
        ]

        for text, expected_count in test_cases:
            matches = pattern.findall(text)
            assert len(matches) == expected_count, f"Failed for: {text}"

    def test_detect_base64_data_uri(self):
        """Test detection of base64 data URIs."""
        import re

        pattern = r"data:image/[a-zA-Z]+;base64,"

        test_cases = [
            ("data:image/jpeg;base64,/9j/4AAQSkZJRg==", 1),
            ("data:image/png;base64,iVBORw0KGgo=", 1),
            ("Multiple: data:image/jpg;base64,abc data:image/png;base64,def", 2),
            ("No images", 0),
            ("data:text/plain;base64,", 0),  # Not an image
        ]

        for text, expected_count in test_cases:
            matches = re.findall(pattern, text)
            assert len(matches) == expected_count, f"Failed for: {text[:50]}"

    def test_multimodal_format_detection(self):
        """Test detection in OpenAI multimodal format."""

        # Message with image_url in content array
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/photo.jpg"},
                },
            ],
        }

        # Simulate detection logic
        image_count = 0
        content = message.get("content", "")

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_count += 1

        assert image_count == 1

    def test_combined_detection(self):
        """Test detection across multiple methods."""
        import re

        url_pattern = re.compile(
            r'https?://[^\s<>"]+\.(?:jpg|jpeg|png|gif|bmp|webp|svg)(?:[?#][^\s<>"]*)?',
            re.IGNORECASE,
        )
        base64_pattern = r"data:image/[a-zA-Z]+;base64,"

        # Text with both URL and base64
        text = (
            "URL: https://example.com/pic.jpg and "
            "base64: data:image/png;base64,iVBORw0KGgo="
        )

        url_count = len(url_pattern.findall(text))
        base64_count = len(re.findall(base64_pattern, text))
        total = url_count + base64_count

        assert total == 2


class TestVisionMetadata:
    """Test vision metadata in responses."""

    def test_vision_metadata_structure(self):
        """Test expected structure of vision metadata."""
        # Expected metadata format from proxy
        metadata = {
            "routing": {
                "provider_id": "openai_cloud",
                "model_id": "gpt-4o",
                "estimated_cost": 0.01,
                "task": "vision",
            },
            "vision": {
                "image_count": 2,
                "supports_vision": True,
            },
        }

        # Validate structure
        assert "vision" in metadata
        assert "image_count" in metadata["vision"]
        assert "supports_vision" in metadata["vision"]
        assert metadata["vision"]["image_count"] == 2
        assert metadata["vision"]["supports_vision"] is True

    def test_no_vision_metadata_for_text_only(self):
        """Test no vision metadata for text-only requests."""
        # Text-only request metadata
        metadata = {
            "routing": {
                "provider_id": "ollama_local",
                "model_id": "llama3.1:8b",
                "estimated_cost": 0.0,
                "task": "chat_small",
            }
        }

        # Should not have vision key
        assert "vision" not in metadata


class TestVisionConfigValidation:
    """Test vision configuration validation."""

    def test_vision_model_with_vision_task(self):
        """Test vision model with vision task is valid."""
        model = ModelConfig(
            model_id="gpt-4o",
            task=ModelTask.VISION,
            supports_vision=True,
        )

        assert model.task == ModelTask.VISION
        assert model.supports_vision is True

    def test_vision_capabilities_limits(self):
        """Test vision capability limits are reasonable."""
        capabilities = VisionCapabilities(
            max_images=20,
            max_image_size_mb=50,
            supported_formats=["jpg", "png", "gif", "webp", "bmp", "svg"],
        )

        assert capabilities.max_images > 0
        assert capabilities.max_image_size_mb > 0
        assert len(capabilities.supported_formats) > 0

    def test_provider_specific_capabilities(self):
        """Test different providers have different capabilities."""
        # OpenAI GPT-4o
        gpt4o = VisionCapabilities(
            max_images=10,
            max_image_size_mb=20,
            supported_formats=["jpg", "jpeg", "png", "gif", "webp"],
        )

        # Anthropic Claude 3
        claude3 = VisionCapabilities(
            max_images=5,
            max_image_size_mb=5,
            supported_formats=["jpg", "jpeg", "png", "gif", "webp"],
        )

        # Google Gemini Pro 1.5
        gemini = VisionCapabilities(
            max_images=16,
            max_image_size_mb=10,
            supported_formats=["jpg", "jpeg", "png", "gif", "webp"],
        )

        # Validate different limits
        assert gpt4o.max_images == 10
        assert claude3.max_images == 5
        assert gemini.max_images == 16

        assert gpt4o.max_image_size_mb == 20
        assert claude3.max_image_size_mb == 5
        assert gemini.max_image_size_mb == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
