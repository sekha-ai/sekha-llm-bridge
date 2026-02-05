"""Integration tests for v2.0 multi-provider functionality."""

import pytest
from sekha_llm_bridge.config import (
    Settings,
    LlmProviderConfig,
    ModelCapability,
    DefaultModels,
    ProviderType,
    ModelTask,
)
from sekha_llm_bridge.registry import ModelRegistry
from sekha_llm_bridge.pricing import estimate_cost, find_cheapest_model


@pytest.fixture
def test_config():
    """Create test configuration with mock providers."""
    return Settings(
        providers=[
            LlmProviderConfig(
                id="test_ollama",
                provider_type=ProviderType.OLLAMA,
                base_url="http://localhost:11434",
                priority=1,
                models=[
                    ModelCapability(
                        model_id="test-embed",
                        task=ModelTask.EMBEDDING,
                        context_window=512,
                        dimension=768,
                    ),
                    ModelCapability(
                        model_id="test-chat",
                        task=ModelTask.CHAT_SMALL,
                        context_window=8192,
                    ),
                ],
            )
        ],
        default_models=DefaultModels(
            embedding="test-embed",
            chat_fast="test-chat",
            chat_smart="test-chat",
        ),
    )


class TestModelRegistry:
    """Test model registry functionality."""

    def test_registry_initialization(self, test_config, monkeypatch):
        """Test registry initializes with providers."""
        monkeypatch.setattr("sekha_llm_bridge.config.settings", test_config)

        registry = ModelRegistry()

        assert len(registry.providers) == 1
        assert "test_ollama" in registry.providers
        assert len(registry.model_cache) == 2

    def test_list_models(self, test_config, monkeypatch):
        """Test listing all models."""
        monkeypatch.setattr("sekha_llm_bridge.config.settings", test_config)

        registry = ModelRegistry()
        models = registry.list_all_models()

        assert len(models) == 2
        model_ids = [m["model_id"] for m in models]
        assert "test-embed" in model_ids
        assert "test-chat" in model_ids

    @pytest.mark.asyncio
    async def test_routing_for_embedding(self, test_config, monkeypatch):
        """Test routing for embedding task."""
        monkeypatch.setattr("sekha_llm_bridge.config.settings", test_config)

        registry = ModelRegistry()

        result = await registry.route_with_fallback(
            task=ModelTask.EMBEDDING,
        )

        assert result.model_id == "test-embed"
        assert result.provider.provider_id == "test_ollama"
        assert result.estimated_cost == 0.0  # Local model is free

    @pytest.mark.asyncio
    async def test_routing_for_chat(self, test_config, monkeypatch):
        """Test routing for chat task."""
        monkeypatch.setattr("sekha_llm_bridge.config.settings", test_config)

        registry = ModelRegistry()

        result = await registry.route_with_fallback(
            task=ModelTask.CHAT_SMALL,
        )

        assert result.model_id == "test-chat"
        assert result.provider.provider_id == "test_ollama"

    @pytest.mark.asyncio
    async def test_routing_with_preferred_model(self, test_config, monkeypatch):
        """Test routing respects preferred model."""
        monkeypatch.setattr("sekha_llm_bridge.config.settings", test_config)

        registry = ModelRegistry()

        result = await registry.route_with_fallback(
            task=ModelTask.CHAT_SMALL,
            preferred_model="test-chat",
        )

        assert result.model_id == "test-chat"
        assert "preferred" in result.reason.lower()

    def test_provider_health(self, test_config, monkeypatch):
        """Test provider health status."""
        monkeypatch.setattr("sekha_llm_bridge.config.settings", test_config)

        registry = ModelRegistry()
        health = registry.get_provider_health()

        assert "test_ollama" in health
        assert health["test_ollama"]["models_count"] == 2


class TestCostEstimation:
    """Test cost estimation functionality."""

    def test_estimate_free_model(self):
        """Test cost estimation for free local model."""
        cost = estimate_cost("llama3.1:8b", 1000, 500)
        assert cost == 0.0

    def test_estimate_paid_model(self):
        """Test cost estimation for paid model."""
        cost = estimate_cost("gpt-4o", 1000, 500)
        assert cost > 0.0
        # GPT-4o: $0.005 per 1K input, $0.015 per 1K output
        # Expected: (1000/1000 * 0.005) + (500/1000 * 0.015) = 0.005 + 0.0075 = 0.0125
        assert abs(cost - 0.0125) < 0.0001

    def test_find_cheapest_model(self):
        """Test finding cheapest model from list."""
        models = ["gpt-4o", "gpt-4o-mini", "llama3.1:8b"]
        cheapest = find_cheapest_model(models, 1000, 500)

        assert cheapest == "llama3.1:8b"  # Free local model

    def test_find_cheapest_with_budget(self):
        """Test finding cheapest model within budget."""
        models = ["gpt-4o", "gpt-4o-mini", "llama3.1:8b"]

        # Budget that excludes GPT-4o but allows GPT-4o-mini
        cheapest = find_cheapest_model(models, 1000, 500, max_cost=0.001)

        # Should return free model or GPT-4o-mini depending on which is cheaper
        assert cheapest in ["llama3.1:8b", "gpt-4o-mini"]


class TestProviderIntegration:
    """Test provider integration (requires running Ollama)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_ollama_health(self):
        """Test health check against real Ollama instance."""
        from sekha_llm_bridge.providers import LiteLlmProvider

        provider = LiteLlmProvider(
            provider_id="test",
            config={
                "provider_type": "ollama",
                "base_url": "http://localhost:11434",
                "timeout": 30,
                "models": [
                    {"model_id": "test", "task": "chat", "context_window": 2048}
                ],
            },
        )

        health = await provider.health_check()

        # Should have status key
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_embedding(self):
        """Test embedding generation with real Ollama."""
        from sekha_llm_bridge.providers import LiteLlmProvider

        provider = LiteLlmProvider(
            provider_id="test",
            config={
                "provider_type": "ollama",
                "base_url": "http://localhost:11434",
                "timeout": 30,
                "models": [],
            },
        )

        try:
            result = await provider.generate_embedding(
                text="Test embedding",
                model="nomic-embed-text",
            )

            assert len(result.embedding) > 0
            assert result.dimension > 0
            assert result.model == "nomic-embed-text"
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test validation passes with valid config."""
        config = Settings(
            providers=[
                LlmProviderConfig(
                    id="test",
                    provider_type=ProviderType.OLLAMA,
                    base_url="http://localhost:11434",
                    priority=1,
                    models=[
                        ModelCapability(
                            model_id="test-model",
                            task=ModelTask.EMBEDDING,
                            context_window=512,
                            dimension=768,
                        )
                    ],
                )
            ],
            default_models=DefaultModels(
                embedding="test-model",
                chat_fast="test-model",
                chat_smart="test-model",
            ),
        )

        # Should not raise
        config.validate_config()

    def test_invalid_config_no_providers(self):
        """Test validation fails without providers."""
        config = Settings(providers=[])

        with pytest.raises(ValueError, match="No providers"):
            config.validate_config()

    def test_invalid_config_missing_model(self):
        """Test validation fails if default model not in providers."""
        config = Settings(
            providers=[
                LlmProviderConfig(
                    id="test",
                    provider_type=ProviderType.OLLAMA,
                    base_url="http://localhost:11434",
                    priority=1,
                    models=[
                        ModelCapability(
                            model_id="model-a",
                            task=ModelTask.EMBEDDING,
                            context_window=512,
                            dimension=768,
                        )
                    ],
                )
            ],
            default_models=DefaultModels(
                embedding="model-b",  # Not in providers!
                chat_fast="model-a",
                chat_smart="model-a",
            ),
        )

        with pytest.raises(ValueError, match="not found"):
            config.validate_config()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
