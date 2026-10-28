from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str
    tavily_api_key: str
    github_token: str

    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2

    http_timeout_seconds: float = 20.0
    http_connect_timeout_seconds: float = 5.0
    http_max_connections: int = 50
    http_max_keepalive_connections: int = 20
    http_max_attempts: int = 3

    tavily_results_per_query: int = 5
    github_results_per_query: int = 10
    min_relevance_score: float = 0.2

    max_concurrent_pipelines: int = 3
    max_concurrent_searches: int = 8
    max_concurrent_llm_calls: int = 4
    max_pending_jobs: int = 20
    pipeline_timeout_seconds: float = 180.0
    job_ttl_seconds: int = 900
    max_stored_jobs: int = 200
    sse_keepalive_seconds: float = 15.0


@lru_cache
def get_settings() -> Settings:
    """Returns the cached application settings.

    Returns:
        Settings loaded from environment variables and backend/.env.
    """
    return Settings()