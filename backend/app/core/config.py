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

    llm_model: str = "llama-3.3-70b-versatile"
    fast_llm_model: str = "llama-3.1-8b-instant"

    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Returns the cached application settings.

    Returns:
        Settings loaded from environment variables and backend/.env.
    """
    return Settings()
