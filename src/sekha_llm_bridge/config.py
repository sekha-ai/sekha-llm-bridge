from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """LLM Bridge configuration"""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 5001
    workers: int = 4
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
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
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
