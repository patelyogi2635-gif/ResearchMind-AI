from functools import lru_cache

from langchain_groq import ChatGroq

from app.core.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.3) -> ChatGroq:
    """Returns the cached primary Groq chat model.

    Args:
        temperature: Sampling temperature. Use low values for structured output.

    Returns:
        A ChatGroq instance configured with the primary model.
    """
    settings = get_settings()
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
    )


@lru_cache
def get_fast_llm() -> ChatGroq:
    """Returns the cached small, fast Groq model used for lightweight tasks."""
    settings = get_settings()
    return ChatGroq(
        model=settings.fast_llm_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
    )
