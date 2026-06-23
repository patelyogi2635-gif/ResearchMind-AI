"""
config.py — single source of truth for all environment variables.
Updated to use Groq instead of Anthropic — free, fast, llama-3.3-70b.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # LLM — Groq (free tier, no credit card)
    groq_api_key: str

    # MCP servers
    tavily_api_key: str
    github_token: str

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Model to use — llama-3.3-70b is free and very capable
    llm_model: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = str(Path(__file__).parent.parent / ".env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()