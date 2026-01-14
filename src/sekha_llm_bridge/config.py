from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """LLM Bridge configuration"""

    # Server
    host: str = "0.0.0.0"
    port: int = 5001
    workers: int = 4

    # Ollama - use Field with validation_alias to accept OLLAMA_URL
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_URL"
    )
    ollama_timeout: int = 120

    # Models
    embedding_model: str = "nomic-embed-text:latest"
    summarization_model: str = "llama3.1:8b"
    extraction_model: str = "llama3.1:8b"

    # Redis (for Celery)
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # LiteLLM (optional cloud providers)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Health check
    health_check_interval: int = 60

    # Logging - use Field with validation_alias to accept LOG_LEVEL
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False
        # Allow population by field name OR alias
        populate_by_name = True


settings = Settings()
