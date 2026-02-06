"""Integration tests for v2 implementation."""

from sekha_llm_bridge.config import (ModelConfig, ModelTask, ProviderConfig,
                                     ProviderType)


class TestProviderConfigIntegration:
    """Test provider configuration integration."""

    def test_provider_config_creation(self):
        """Test creating a provider config."""
        config = ProviderConfig(
            id="test-provider",
            provider_type=ProviderType.LITELLM,
            base_url="http://localhost:8000",
            models=[],
        )
        assert config.id == "test-provider"
        assert config.provider_type == ProviderType.LITELLM

    def test_model_config_creation(self):
        """Test creating a model config."""
        model = ModelConfig(
            model_id="gpt-4o", task=ModelTask.CHAT_SMART, context_window=128000
        )
        assert model.model_id == "gpt-4o"
        assert model.task == ModelTask.CHAT_SMART

    def test_full_provider_with_models(self):
        """Test creating a provider with multiple models."""
        models = [
            ModelConfig(
                model_id="gpt-4o", task=ModelTask.CHAT_SMART, context_window=128000
            ),
            ModelConfig(
                model_id="gpt-4o-mini", task=ModelTask.CHAT_SMALL, context_window=128000
            ),
        ]

        config = ProviderConfig(
            id="openai",
            provider_type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key="${OPENAI_API_KEY}",
            models=models,
        )

        assert len(config.models) == 2
        assert config.models[0].model_id == "gpt-4o"
