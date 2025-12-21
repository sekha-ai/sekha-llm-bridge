from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service
    host: str = "0.0.0.0"
    port: int = 5001

    # Models
    default_embed_model: str = "nomic-embed-text"
    default_summarize_model: str = "ollama/llama3.1"
    default_importance_model: str = "ollama/llama3.1"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Celery / Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    class Config:
        env_prefix = "SEKHA_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
