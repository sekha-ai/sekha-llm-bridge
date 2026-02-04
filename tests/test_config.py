"""Unit tests for configuration management."""

import pytest
import os
from sekha_llm_bridge.config import (
    Settings,
    LlmProviderConfig,
    ModelCapability,
    DefaultModels,
    ProviderType,
    ModelTask,
    RoutingConfig,
    CircuitBreakerConfig,
)


class TestProviderConfiguration:
    """Test provider configuration models."""
    
    def test_provider_type_enum(self):
        """Test provider type enum values."""
        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
    
    def test_model_task_enum(self):
        """Test model task enum values."""
        assert ModelTask.EMBEDDING.value == "embedding"
        assert ModelTask.CHAT_SMALL.value == "chat_small"
        assert ModelTask.CHAT_SMART.value == "chat_smart"
        assert ModelTask.VISION.value == "vision"
    
    def test_model_capability_creation(self):
        """Test creating a model capability."""
        model = ModelCapability(
            model_id="nomic-embed-text",
            task=ModelTask.EMBEDDING,
            context_window=512,
            dimension=768,
        )
        
        assert model.model_id == "nomic-embed-text"
        assert model.task == ModelTask.EMBEDDING
        assert model.dimension == 768
        assert not model.supports_vision
    
    def test_provider_config_creation(self):
        """Test creating a provider configuration."""
        provider = LlmProviderConfig(
            id="ollama_local",
            type=ProviderType.OLLAMA,
            base_url="http://localhost:11434",
            priority=1,
            models=[
                ModelCapability(
                    model_id="llama3.1:8b",
                    task=ModelTask.CHAT_SMALL,
                    context_window=8192,
                )
            ],
        )
        
        assert provider.id == "ollama_local"
        assert provider.provider_type == ProviderType.OLLAMA
        assert len(provider.models) == 1


class TestDefaultModels:
    """Test default model configuration."""
    
    def test_default_models_creation(self):
        """Test creating default models config."""
        defaults = DefaultModels(
            embedding="nomic-embed-text",
            chat_fast="llama3.1:8b",
            chat_smart="gpt-4o",
            chat_vision="gpt-4o",
        )
        
        assert defaults.embedding == "nomic-embed-text"
        assert defaults.chat_fast == "llama3.1:8b"
        assert defaults.chat_smart == "gpt-4o"
        assert defaults.chat_vision == "gpt-4o"
    
    def test_default_models_optional_vision(self):
        """Test that vision model is optional."""
        defaults = DefaultModels(
            embedding="nomic-embed-text",
            chat_fast="llama3.1:8b",
            chat_smart="llama3.1:8b",
        )
        
        assert defaults.chat_vision is None


class TestRoutingConfig:
    """Test routing configuration."""
    
    def test_routing_config_defaults(self):
        """Test default routing configuration."""
        routing = RoutingConfig()
        
        assert routing.auto_fallback is True
        assert routing.require_vision_for_images is True
        assert routing.max_cost_per_request is None
        assert routing.circuit_breaker.failure_threshold == 3
    
    def test_circuit_breaker_config(self):
        """Test circuit breaker configuration."""
        cb = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_secs=120,
            success_threshold=3,
        )
        
        assert cb.failure_threshold == 5
        assert cb.timeout_secs == 120
        assert cb.success_threshold == 3


class TestSettingsValidation:
    """Test settings validation."""
    
    def test_validate_with_valid_config(self):
        """Test validation passes with valid configuration."""
        settings = Settings(
            providers=[
                LlmProviderConfig(
                    id="ollama_test",
                    type=ProviderType.OLLAMA,
                    base_url="http://localhost:11434",
                    priority=1,
                    models=[
                        ModelCapability(
                            model_id="nomic-embed-text",
                            task=ModelTask.EMBEDDING,
                            context_window=512,
                            dimension=768,
                        ),
                        ModelCapability(
                            model_id="llama3.1:8b",
                            task=ModelTask.CHAT_SMALL,
                            context_window=8192,
                        ),
                    ],
                )
            ],
            default_models=DefaultModels(
                embedding="nomic-embed-text",
                chat_fast="llama3.1:8b",
                chat_smart="llama3.1:8b",
            ),
        )
        
        # Should not raise
        settings.validate_config()
    
    def test_validate_fails_with_no_providers(self):
        """Test validation fails without providers."""
        settings = Settings(providers=[])
        
        with pytest.raises(ValueError, match="No providers configured"):
            settings.validate_config()
    
    def test_validate_fails_with_missing_default_models(self):
        """Test validation fails without default models."""
        settings = Settings(
            providers=[
                LlmProviderConfig(
                    id="test",
                    type=ProviderType.OLLAMA,
                    base_url="http://localhost:11434",
                    priority=1,
                )
            ],
            default_models=None,
        )
        
        with pytest.raises(ValueError, match="default_models must be specified"):
            settings.validate_config()
    
    def test_validate_fails_with_nonexistent_embedding_model(self):
        """Test validation fails if embedding model not in providers."""
        settings = Settings(
            providers=[
                LlmProviderConfig(
                    id="test",
                    type=ProviderType.OLLAMA,
                    base_url="http://localhost:11434",
                    priority=1,
                    models=[
                        ModelCapability(
                            model_id="llama3.1:8b",
                            task=ModelTask.CHAT_SMALL,
                            context_window=8192,
                        )
                    ],
                )
            ],
            default_models=DefaultModels(
                embedding="nonexistent-model",
                chat_fast="llama3.1:8b",
                chat_smart="llama3.1:8b",
            ),
        )
        
        with pytest.raises(ValueError, match="Default embedding model.*not found"):
            settings.validate_config()
    
    def test_validate_fails_with_duplicate_provider_ids(self):
        """Test validation fails with duplicate provider IDs."""
        settings = Settings(
            providers=[
                LlmProviderConfig(
                    id="duplicate",
                    type=ProviderType.OLLAMA,
                    base_url="http://localhost:11434",
                    priority=1,
                    models=[
                        ModelCapability(
                            model_id="model1",
                            task=ModelTask.CHAT_SMALL,
                            context_window=8192,
                        )
                    ],
                ),
                LlmProviderConfig(
                    id="duplicate",  # Same ID
                    type=ProviderType.OPENAI,
                    base_url="https://api.openai.com/v1",
                    priority=2,
                    models=[
                        ModelCapability(
                            model_id="gpt-4o",
                            task=ModelTask.CHAT_SMART,
                            context_window=128000,
                        )
                    ],
                ),
            ],
            default_models=DefaultModels(
                embedding="model1",
                chat_fast="model1",
                chat_smart="model1",
            ),
        )
        
        with pytest.raises(ValueError, match="Duplicate provider IDs"):
            settings.validate_config()


class TestMigration:
    """Test v1.x to v2.0 auto-migration."""
    
    def test_auto_migration_from_v1(self, monkeypatch):
        """Test automatic migration from v1.x configuration."""
        # Set v1.x style environment variables
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama:11434")
        monkeypatch.setenv("EMBEDDING_MODEL", "test-embed")
        monkeypatch.setenv("SUMMARIZATION_MODEL", "test-chat")
        
        # Create settings without explicit providers
        settings = Settings(
            ollama_base_url="http://test-ollama:11434",
            embedding_model="test-embed",
            summarization_model="test-chat",
        )
        
        # Should have auto-migrated
        assert len(settings.providers) == 1
        assert settings.providers[0].id == "ollama_migrated"
        assert settings.providers[0].base_url == "http://test-ollama:11434"
        
        # Should have created default models
        assert settings.default_models is not None
        assert settings.default_models.embedding == "test-embed"
        assert settings.default_models.chat_fast == "test-chat"
    
    def test_no_migration_with_v2_config(self):
        """Test that v2.0 config is not modified by migration."""
        settings = Settings(
            providers=[
                LlmProviderConfig(
                    id="custom_provider",
                    type=ProviderType.OPENAI,
                    base_url="https://api.openai.com/v1",
                    priority=1,
                    models=[
                        ModelCapability(
                            model_id="gpt-4o",
                            task=ModelTask.CHAT_SMART,
                            context_window=128000,
                        )
                    ],
                )
            ],
            default_models=DefaultModels(
                embedding="text-embedding-3-large",
                chat_fast="gpt-4o-mini",
                chat_smart="gpt-4o",
            ),
            # Old config present but should be ignored
            ollama_base_url="http://localhost:11434",
        )
        
        # Should keep v2.0 config
        assert len(settings.providers) == 1
        assert settings.providers[0].id == "custom_provider"
        assert settings.default_models.embedding == "text-embedding-3-large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
