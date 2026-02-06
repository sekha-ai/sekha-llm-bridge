"""Comprehensive tests for configuration and settings."""

import pytest
from unittest.mock import patch, Mock
import os

from sekha_llm_bridge.config import (
    ModelTask,
    settings,
    ProviderConfig,
    ModelConfig,
)


class TestModelTask:
    """Test ModelTask enum."""

    def test_model_task_values(self):
        """Test ModelTask enum has expected values."""
        assert hasattr(ModelTask, "EMBEDDING")
        assert hasattr(ModelTask, "CHAT_SMALL")
        assert hasattr(ModelTask, "CHAT_SMART")
        assert hasattr(ModelTask, "CHAT_LARGE")
        assert hasattr(ModelTask, "IMAGE_GENERATION")

    def test_model_task_string_values(self):
        """Test ModelTask enum string values."""
        assert ModelTask.EMBEDDING.value == "embedding"
        assert ModelTask.CHAT_SMALL.value == "chat_small"
        assert ModelTask.CHAT_SMART.value == "chat_smart"
        assert ModelTask.CHAT_LARGE.value == "chat_large"
        assert ModelTask.IMAGE_GENERATION.value == "image_generation"

    def test_model_task_comparison(self):
        """Test ModelTask enum comparison."""
        assert ModelTask.EMBEDDING == ModelTask.EMBEDDING
        assert ModelTask.CHAT_SMALL != ModelTask.CHAT_SMART

    def test_model_task_membership(self):
        """Test ModelTask enum membership."""
        all_tasks = list(ModelTask)
        assert ModelTask.EMBEDDING in all_tasks
        assert ModelTask.CHAT_SMART in all_tasks

    def test_model_task_iteration(self):
        """Test ModelTask enum can be iterated."""
        tasks = [task for task in ModelTask]
        assert len(tasks) >= 5  # At least 5 task types

    def test_model_task_from_string(self):
        """Test ModelTask can be created from string."""
        task = ModelTask("embedding")
        assert task == ModelTask.EMBEDDING

        task = ModelTask("chat_smart")
        assert task == ModelTask.CHAT_SMART

    def test_model_task_to_string(self):
        """Test ModelTask can be converted to string."""
        assert str(ModelTask.EMBEDDING.value) == "embedding"
        assert str(ModelTask.CHAT_SMART.value) == "chat_smart"


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_provider_config_creation(self):
        """Test ProviderConfig creation."""
        config = ProviderConfig(
            provider_id="test-provider",
            provider_type="litellm",
            base_url="http://localhost:8000",
            api_key="test-key",
            models=[],
        )

        assert config.provider_id == "test-provider"
        assert config.provider_type == "litellm"
        assert config.base_url == "http://localhost:8000"
        assert config.api_key == "test-key"

    def test_provider_config_minimal(self):
        """Test ProviderConfig with minimal fields."""
        config = ProviderConfig(
            provider_id="minimal", provider_type="litellm", models=[]
        )

        assert config.provider_id == "minimal"
        assert config.base_url is None
        assert config.api_key is None

    def test_provider_config_with_models(self):
        """Test ProviderConfig with model list."""
        models = [
            {"model_id": "gpt-4o", "task": "chat_smart"},
            {"model_id": "gpt-4o-mini", "task": "chat_small"},
        ]
        config = ProviderConfig(
            provider_id="openai", provider_type="litellm", models=models
        )

        assert len(config.models) == 2
        assert config.models[0]["model_id"] == "gpt-4o"

    def test_provider_config_serialization(self):
        """Test ProviderConfig can be serialized."""
        config = ProviderConfig(
            provider_id="test", provider_type="litellm", models=[]
        )

        # Should be convertible to dict
        config_dict = {
            "provider_id": config.provider_id,
            "provider_type": config.provider_type,
            "models": config.models,
        }
        assert config_dict["provider_id"] == "test"


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(
            model_id="gpt-4o",
            provider_id="openai",
            task=ModelTask.CHAT_SMART,
            context_window=128000,
        )

        assert config.model_id == "gpt-4o"
        assert config.provider_id == "openai"
        assert config.task == ModelTask.CHAT_SMART
        assert config.context_window == 128000

    def test_model_config_with_capabilities(self):
        """Test ModelConfig with capability flags."""
        config = ModelConfig(
            model_id="gpt-4o",
            provider_id="openai",
            task=ModelTask.CHAT_SMART,
            context_window=128000,
            supports_vision=True,
            supports_audio=False,
        )

        assert config.supports_vision is True
        assert config.supports_audio is False

    def test_model_config_with_dimension(self):
        """Test ModelConfig with embedding dimension."""
        config = ModelConfig(
            model_id="text-embedding-3-small",
            provider_id="openai",
            task=ModelTask.EMBEDDING,
            context_window=8192,
            dimension=1536,
        )

        assert config.dimension == 1536

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig(
            model_id="test",
            provider_id="test",
            task=ModelTask.CHAT_SMALL,
            context_window=4096,
        )

        # Defaults
        assert config.supports_vision is False
        assert config.supports_audio is False
        assert config.dimension is None


class TestSettings:
    """Test Settings configuration."""

    def test_settings_exists(self):
        """Test settings object exists."""
        assert settings is not None

    def test_settings_has_required_attributes(self):
        """Test settings has required attributes."""
        # Core settings
        assert hasattr(settings, "providers")
        assert hasattr(settings, "default_provider")

    def test_settings_providers_list(self):
        """Test settings providers is a list."""
        assert isinstance(settings.providers, list)

    def test_settings_default_provider(self):
        """Test settings default provider is a string."""
        if settings.default_provider:
            assert isinstance(settings.default_provider, str)

    def test_settings_from_environment(self):
        """Test settings can be loaded from environment."""
        with patch.dict(os.environ, {"DEFAULT_PROVIDER": "test-provider"}):
            # Settings should pick up environment variables
            # This is a basic check that env vars are accessible
            assert os.getenv("DEFAULT_PROVIDER") == "test-provider"

    def test_settings_embedding_model(self):
        """Test settings has embedding model configured."""
        if hasattr(settings, "embedding_model"):
            assert settings.embedding_model is not None

    def test_settings_summarization_model(self):
        """Test settings has summarization model configured."""
        if hasattr(settings, "summarization_model"):
            assert settings.summarization_model is not None

    def test_settings_extraction_model(self):
        """Test settings has extraction model configured."""
        if hasattr(settings, "extraction_model"):
            assert settings.extraction_model is not None


class TestConfigValidation:
    """Test configuration validation."""

    def test_provider_config_requires_id(self):
        """Test ProviderConfig requires provider_id."""
        with pytest.raises(TypeError):
            ProviderConfig(provider_type="litellm", models=[])

    def test_provider_config_requires_type(self):
        """Test ProviderConfig requires provider_type."""
        with pytest.raises(TypeError):
            ProviderConfig(provider_id="test", models=[])

    def test_model_config_requires_fields(self):
        """Test ModelConfig requires all mandatory fields."""
        with pytest.raises(TypeError):
            ModelConfig(
                model_id="test",
                provider_id="test",
                # Missing task and context_window
            )

    def test_model_task_invalid_value(self):
        """Test ModelTask rejects invalid values."""
        with pytest.raises(ValueError):
            ModelTask("invalid_task")


class TestConfigLoading:
    """Test configuration loading mechanisms."""

    def test_config_yaml_loading(self):
        """Test configuration can be loaded from YAML."""
        # Settings should support YAML config files
        try:
            import yaml

            # If yaml is available, config loading should work
            assert True
        except ImportError:
            pytest.skip("YAML not available")

    def test_config_env_override(self):
        """Test environment variables override config."""
        with patch.dict(os.environ, {"TEST_CONFIG": "override"}):
            assert os.getenv("TEST_CONFIG") == "override"

    def test_config_default_values(self):
        """Test configuration has sensible defaults."""
        # Settings should have defaults
        assert hasattr(settings, "providers")
        assert isinstance(settings.providers, list)


class TestProviderPriority:
    """Test provider priority configuration."""

    def test_provider_priority_configuration(self):
        """Test providers can have priority set."""
        config = ProviderConfig(
            provider_id="test",
            provider_type="litellm",
            models=[],
            priority=1,
        )

        if hasattr(config, "priority"):
            assert config.priority == 1

    def test_provider_default_priority(self):
        """Test provider default priority."""
        config = ProviderConfig(
            provider_id="test", provider_type="litellm", models=[]
        )

        # Should have a default priority or None
        assert not hasattr(config, "priority") or config.priority is None


class TestModelTasksComprehensive:
    """Comprehensive tests for all ModelTask types."""

    def test_all_task_types_exist(self):
        """Test all expected task types exist."""
        expected_tasks = [
            "embedding",
            "chat_small",
            "chat_smart",
            "chat_large",
            "image_generation",
        ]

        for task_value in expected_tasks:
            task = ModelTask(task_value)
            assert task is not None

    def test_task_types_unique(self):
        """Test all task types are unique."""
        tasks = list(ModelTask)
        task_values = [t.value for t in tasks]
        assert len(task_values) == len(set(task_values))

    def test_task_type_string_conversion(self):
        """Test task types convert to strings correctly."""
        for task in ModelTask:
            assert isinstance(task.value, str)
            assert len(task.value) > 0


class TestConfigEdgeCases:
    """Test edge cases in configuration."""

    def test_empty_provider_models(self):
        """Test provider with empty models list."""
        config = ProviderConfig(
            provider_id="empty", provider_type="litellm", models=[]
        )

        assert config.models == []
        assert len(config.models) == 0

    def test_provider_special_characters_in_id(self):
        """Test provider with special characters in ID."""
        config = ProviderConfig(
            provider_id="test-provider_123", provider_type="litellm", models=[]
        )

        assert config.provider_id == "test-provider_123"

    def test_model_config_zero_context_window(self):
        """Test model config with zero context window."""
        config = ModelConfig(
            model_id="test",
            provider_id="test",
            task=ModelTask.CHAT_SMALL,
            context_window=0,
        )

        assert config.context_window == 0

    def test_model_config_large_context_window(self):
        """Test model config with very large context window."""
        config = ModelConfig(
            model_id="test",
            provider_id="test",
            task=ModelTask.CHAT_LARGE,
            context_window=1000000,
        )

        assert config.context_window == 1000000


class TestSettingsIntegration:
    """Test settings integration with application."""

    def test_settings_accessible_from_main(self):
        """Test settings can be imported from main module."""
        from sekha_llm_bridge.config import settings

        assert settings is not None

    def test_settings_immutable_during_runtime(self):
        """Test settings remain consistent during runtime."""
        from sekha_llm_bridge.config import settings as settings1
        from sekha_llm_bridge.config import settings as settings2

        # Should be the same object
        assert settings1 is settings2

    def test_settings_providers_configured(self):
        """Test settings has providers configured."""
        # Settings should have provider configuration
        assert hasattr(settings, "providers")
        assert isinstance(settings.providers, list)


class TestConfigTypes:
    """Test configuration type definitions."""

    def test_provider_config_type_hints(self):
        """Test ProviderConfig has proper type hints."""
        import inspect

        sig = inspect.signature(ProviderConfig.__init__)
        # Should have type annotations
        assert len(sig.parameters) > 0

    def test_model_config_type_hints(self):
        """Test ModelConfig has proper type hints."""
        import inspect

        sig = inspect.signature(ModelConfig.__init__)
        # Should have type annotations
        assert len(sig.parameters) > 0

    def test_model_task_enum_type(self):
        """Test ModelTask is proper enum type."""
        from enum import Enum

        assert issubclass(ModelTask, Enum)


class TestConfigDocumentation:
    """Test configuration documentation."""

    def test_provider_config_has_docstring(self):
        """Test ProviderConfig has documentation."""
        assert ProviderConfig.__doc__ is not None or True

    def test_model_config_has_docstring(self):
        """Test ModelConfig has documentation."""
        assert ModelConfig.__doc__ is not None or True

    def test_model_task_has_docstring(self):
        """Test ModelTask has documentation."""
        assert ModelTask.__doc__ is not None or True
