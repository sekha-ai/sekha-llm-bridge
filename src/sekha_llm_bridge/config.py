"""Configuration management for Sekha LLM Bridge."""

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from typing import Optional, List
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    LITELLM = "litellm"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelTask(str, Enum):
    """Task categories for model routing."""
    EMBEDDING = "embedding"
    CHAT_SMALL = "chat_small"
    CHAT_LARGE = "chat_large"
    CHAT_SMART = "chat_smart"
    VISION = "vision"
    AUDIO = "audio"


class ModelCapability(BaseSettings):
    """Metadata about a model's capabilities."""
    model_config = ConfigDict(extra='forbid')
    
    model_id: str
    task: ModelTask
    context_window: int
    supports_vision: bool = False
    supports_audio: bool = False
    dimension: Optional[int] = None  # For embedding models


class LlmProviderConfig(BaseSettings):
    """Configuration for a single LLM provider."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    provider_type: ProviderType = Field(..., alias="type")
    base_url: str
    api_key: Optional[str] = None
    timeout_secs: int = Field(default=120, alias="timeout")
    priority: int = 1
    models: List[ModelCapability] = Field(default_factory=list)


class DefaultModels(BaseSettings):
    """Default model selections for common tasks."""
    model_config = ConfigDict(extra='forbid')
    
    embedding: str
    chat_fast: str
    chat_smart: str
    chat_vision: Optional[str] = None


class CircuitBreakerConfig(BaseSettings):
    """Circuit breaker configuration."""
    model_config = ConfigDict(extra='forbid')
    
    failure_threshold: int = 3
    timeout_secs: int = 60
    success_threshold: int = 2


class RoutingConfig(BaseSettings):
    """Routing and fallback configuration."""
    model_config = ConfigDict(extra='forbid')
    
    auto_fallback: bool = True
    require_vision_for_images: bool = True
    max_cost_per_request: Optional[float] = None
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


class Settings(BaseSettings):
    """LLM Bridge configuration with v2.0 multi-provider support."""
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        populate_by_name=True,
        extra='allow'  # Allow extra fields for flexibility
    )

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 5001
    workers: int = 4

    # ==== V2.0 CONFIGURATION ====
    config_version: Optional[str] = None
    
    # Provider registry (v2.0)
    providers: List[LlmProviderConfig] = Field(
        default_factory=list,
        validation_alias="llm_providers"
    )
    
    # Default model selections (v2.0)
    default_models: Optional[DefaultModels] = None
    
    # Routing configuration (v2.0)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    # ==== DEPRECATED (v1.x) - Keep for backward compatibility ====
    # These will be used for auto-migration if v2.0 config is not present
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_URL"
    )
    embedding_model: str = "nomic-embed-text"
    summarization_model: str = "llama3.1:8b"
    extraction_model: str = "llama3.1:8b"

    # LiteLLM settings
    litellm_verbose: bool = False
    litellm_drop_params: bool = True

    # Redis/Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Cloud provider API keys (optional)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # Monitoring
    health_check_interval: int = 60
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @field_validator("providers", mode="before")
    @classmethod
    def migrate_old_config(cls, v, info):
        """Auto-migrate v1.x configuration to v2.0 format."""
        # If providers list is empty or None, check if we should auto-migrate
        if not v:
            # Get other field values from validation context
            data = info.data
            
            # Check if old-style config exists
            ollama_url = data.get("ollama_base_url", "http://localhost:11434")
            embedding_model = data.get("embedding_model", "nomic-embed-text")
            summarization_model = data.get("summarization_model", "llama3.1:8b")
            
            # Only auto-migrate if we have old config and no new config
            if ollama_url and not v:
                logger.warning(
                    "⚠️  Detected v1.x configuration. Auto-migrating to v2.0 format..."
                )
                
                # Create default Ollama provider from legacy config
                migrated_provider = {
                    "id": "ollama_migrated",
                    "type": "ollama",
                    "base_url": ollama_url,
                    "api_key": None,
                    "priority": 1,
                    "models": [
                        {
                            "model_id": embedding_model,
                            "task": "embedding",
                            "context_window": 512,
                            "dimension": 768,
                        },
                        {
                            "model_id": summarization_model,
                            "task": "chat_small",
                            "context_window": 8192,
                        },
                        {
                            "model_id": summarization_model,
                            "task": "chat_smart",
                            "context_window": 8192,
                        },
                    ],
                }
                
                logger.info(
                    "✅ Auto-migrated Ollama configuration. "
                    "Please update to v2.0 config format for additional providers."
                )
                
                return [migrated_provider]
        
        return v or []

    @field_validator("default_models", mode="before")
    @classmethod
    def set_default_models(cls, v, info):
        """Set default models if not specified but providers exist."""
        if v is None:
            data = info.data
            providers = data.get("providers", [])
            
            # If we have providers but no default models, infer them
            if providers:
                embedding_model = data.get("embedding_model", "nomic-embed-text")
                summarization_model = data.get("summarization_model", "llama3.1:8b")
                
                return {
                    "embedding": embedding_model,
                    "chat_fast": summarization_model,
                    "chat_smart": summarization_model,
                }
        
        return v

    def validate_config(self) -> None:
        """Validate the configuration after loading."""
        if not self.providers:
            raise ValueError(
                "No providers configured. Add at least one provider to llm_providers."
            )
        
        if not self.default_models:
            raise ValueError(
                "default_models must be specified when using providers."
            )
        
        # Validate that default models exist in some provider
        all_model_ids = [
            model.model_id
            for provider in self.providers
            for model in provider.models
        ]
        
        if self.default_models.embedding not in all_model_ids:
            raise ValueError(
                f"Default embedding model '{self.default_models.embedding}' "
                f"not found in any provider"
            )
        
        if self.default_models.chat_fast not in all_model_ids:
            raise ValueError(
                f"Default chat_fast model '{self.default_models.chat_fast}' "
                f"not found in any provider"
            )
        
        if self.default_models.chat_smart not in all_model_ids:
            raise ValueError(
                f"Default chat_smart model '{self.default_models.chat_smart}' "
                f"not found in any provider"
            )
        
        # Check for duplicate provider IDs
        provider_ids = [p.id for p in self.providers]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("Duplicate provider IDs found in configuration")
        
        logger.info(f"✅ Configuration validated: {len(self.providers)} provider(s) configured")


# Global settings instance
try:
    settings = Settings()
    # Validate after loading
    if settings.providers:  # Only validate if v2.0 config exists
        settings.validate_config()
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    # Fall back to defaults for development
    settings = Settings(
        providers=[
            LlmProviderConfig(
                id="ollama_default",
                provider_type=ProviderType.OLLAMA,
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
    logger.warning("⚠️  Using default configuration")
