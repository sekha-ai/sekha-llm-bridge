"""Configuration models for Sekha LLM Bridge."""

import os
from enum import Enum
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, validator


class ProviderType(str, Enum):
    """Supported provider types."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LITELLM = "litellm"


class ModelTask(str, Enum):
    """Model task types for routing."""

    EMBEDDING = "embedding"
    CHAT_SMALL = "chat_small"
    CHAT_LARGE = "chat_large"
    CHAT_SMART = "chat_smart"
    VISION = "vision"
    AUDIO = "audio"
    IMAGE_GENERATION = "image_generation"


class VisionCapabilities(BaseModel):
    """Vision-specific capabilities and limits."""

    max_images: int = Field(default=10, description="Maximum images per request")
    max_image_size_mb: int = Field(default=20, description="Maximum image size in MB")
    supported_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "gif", "webp"],
        description="Supported image formats",
    )


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    model_id: str = Field(..., description="Model identifier (e.g., 'gpt-4o')")
    task: ModelTask = Field(..., description="Primary task this model is optimized for")
    context_window: int = Field(
        default=8192, description="Context window size in tokens"
    )
    dimension: Optional[int] = Field(
        None, description="Embedding dimension (for embedding models only)"
    )
    supports_vision: bool = Field(
        default=False, description="Whether model supports vision/multimodal input"
    )
    supports_audio: bool = Field(
        default=False, description="Whether model supports audio input"
    )
    supports_function_calling: bool = Field(
        default=False, description="Whether model supports function calling"
    )
    vision_capabilities: Optional[VisionCapabilities] = Field(
        None, description="Vision-specific capabilities (if supports_vision=True)"
    )

    @validator("vision_capabilities", always=True)
    def set_vision_defaults(cls, v, values):
        """Set default vision capabilities if model supports vision."""
        if values.get("supports_vision") and v is None:
            return VisionCapabilities()
        return v

    @validator("dimension")
    def validate_dimension(cls, v, values):
        """Validate that dimension is set for embedding models."""
        if values.get("task") == ModelTask.EMBEDDING and v is None:
            raise ValueError("dimension is required for embedding models")
        return v


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    id: str = Field(..., description="Unique provider identifier")
    provider_type: ProviderType = Field(..., description="Type of provider")
    base_url: str = Field(..., description="Base URL for provider API")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    priority: int = Field(
        default=1,
        description="Provider priority (lower number = higher priority)",
        ge=1,
        le=100,
    )
    timeout_secs: int = Field(default=120, description="Request timeout in seconds")
    models: List[ModelConfig] = Field(
        ..., description="List of models available from this provider"
    )

    @validator("api_key", always=True)
    def expand_env_vars(cls, v):
        """Expand environment variables in API key (e.g., ${OPENAI_API_KEY})."""
        if v and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.getenv(env_var)
        return v


class DefaultModels(BaseModel):
    """Default model preferences."""

    embedding: Optional[str] = Field(
        default=None, description="Default embedding model"
    )
    chat_fast: Optional[str] = Field(
        default=None, description="Fast chat model (cheap)"
    )
    chat_smart: Optional[str] = Field(
        default=None, description="Smart chat model (expensive)"
    )
    chat_vision: Optional[str] = Field(
        default=None, description="Vision-capable chat model"
    )


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = Field(
        default=3, description="Failures before opening circuit"
    )
    timeout_secs: int = Field(
        default=60, description="Time to wait before trying again"
    )
    success_threshold: int = Field(
        default=2, description="Successes needed to close circuit"
    )


class RoutingConfig(BaseModel):
    """Request routing configuration."""

    auto_fallback: bool = Field(
        default=True, description="Automatically fallback to other providers on failure"
    )
    require_vision_for_images: bool = Field(
        default=True,
        description="Require vision-capable models when images are detected",
    )
    max_cost_per_request: Optional[float] = Field(
        default=None, description="Maximum cost per request in USD"
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=lambda: CircuitBreakerConfig(),
        description="Circuit breaker settings",
    )


class Settings(BaseModel):
    """Complete application settings."""

    version: str = Field(default="2.0", description="Configuration version")
    providers: List[ProviderConfig] = Field(
        ..., description="List of configured providers"
    )
    default_models: DefaultModels = Field(
        default_factory=lambda: DefaultModels(), description="Default model preferences"
    )
    routing: RoutingConfig = Field(
        default_factory=lambda: RoutingConfig(), description="Routing configuration"
    )

    # Server settings
    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=5001, description="Server port")
    max_connections: int = Field(default=10, description="Max concurrent connections")
    log_level: str = Field(default="info", description="Logging level")

    # Celery/Worker settings
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL for background tasks",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", description="Celery result backend URL"
    )

    @validator("providers")
    def validate_providers(cls, v):
        """Validate that at least one provider is configured."""
        if not v:
            raise ValueError("At least one provider must be configured")
        return v

    @classmethod
    def from_yaml(cls, config_path: str) -> "Settings":
        """Load settings from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables with Ollama defaults.

        This provides a zero-configuration startup for Ollama users.
        Environment variables:
        - OLLAMA_BASE_URL: Ollama server URL (default: http://ollama:11434)
        - EMBEDDING_MODEL: Embedding model name (default: nomic-embed-text)
        - CHAT_MODEL: Chat model name (default: llama3.1:8b)
        - VISION_MODEL: Vision model name (optional)
        - LOG_LEVEL: Log level (default: INFO)
        """
        # Get Ollama URL (Docker-friendly default)
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

        # Get model names
        embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        chat_model = os.getenv("CHAT_MODEL", "llama3.1:8b")
        vision_model = os.getenv("VISION_MODEL", "")  # Optional

        # Build provider config
        models = [
            ModelConfig(
                model_id=embed_model,
                task=ModelTask.EMBEDDING,
                context_window=8192,
                dimension=768,  # nomic-embed-text dimension
            ),
            ModelConfig(
                model_id=chat_model,
                task=ModelTask.CHAT_SMALL,
                context_window=8192,
            ),
            ModelConfig(
                model_id=chat_model,
                task=ModelTask.CHAT_SMART,
                context_window=8192,
            ),
        ]

        # Add vision model if specified
        if vision_model and vision_model.strip() and vision_model.lower() != "none":
            models.append(
                ModelConfig(
                    model_id=vision_model,
                    task=ModelTask.VISION,
                    context_window=8192,
                    supports_vision=True,
                )
            )

        providers = [
            ProviderConfig(
                id="ollama",
                provider_type=ProviderType.OLLAMA,
                base_url=ollama_url,
                priority=1,
                timeout_secs=120,
                models=models,
            )
        ]

        # Build default models
        default_models = DefaultModels(
            embedding=embed_model,
            chat_fast=chat_model,
            chat_smart=chat_model,
            chat_vision=vision_model if vision_model else None,
        )

        # Build settings
        return cls(
            version="2.0",
            providers=providers,
            default_models=default_models,
            routing=RoutingConfig(),
            server_host=os.getenv("HOST", "0.0.0.0"),
            server_port=int(os.getenv("PORT", "5001")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            celery_broker_url=os.getenv(
                "CELERY_BROKER_URL", "redis://localhost:6379/0"
            ),
            celery_result_backend=os.getenv(
                "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
            ),
        )


# Global settings instance - auto-initialize from environment
settings: Optional[Settings] = None


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load and validate settings from config file or environment.

    Args:
        config_path: Path to YAML config file. If None, loads from environment.

    Returns:
        Loaded Settings object
    """
    global settings

    if config_path and os.path.exists(config_path):
        settings = Settings.from_yaml(config_path)
    else:
        # Fall back to environment-based configuration
        settings = Settings.from_env()

    return settings


def get_settings() -> Settings:
    """Get current settings instance."""
    if settings is None:
        # Auto-load from environment if not already loaded
        load_settings()
    return settings


# Auto-initialize settings on module import for zero-config startup
try:
    # Check if config.yaml exists, otherwise use environment
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    if os.path.exists(config_path):
        settings = Settings.from_yaml(config_path)
    else:
        settings = Settings.from_env()
except Exception:
    # Defer initialization to first get_settings() call
    pass
