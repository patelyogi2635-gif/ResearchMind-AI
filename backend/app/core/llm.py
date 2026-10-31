from functools import lru_cache

from langchain_groq import ChatGroq

from app.core.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.3, max_tokens: int | None = None) -> ChatGroq:
    """Returns a cached Groq chat model with timeout and retry settings applied.

    Instances are cached per (temperature, max_tokens) pair, so every request
    reuses the same client and its connection pool.

    Args:
        temperature: Sampling temperature. Use low values for structured output.
        max_tokens: Completion token limit. None uses the provider default.

    Returns:
        A ChatGroq instance configured with the primary model.
    """
    settings = get_settings()
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


@lru_cache
def get_router_llm() -> ChatGroq:
    """Returns the cached small model used to route and screen incoming messages.

    Uses temperature 0 and no internal retries, because the router controller
    owns the retry policy and fails closed.

    Returns:
        A ChatGroq instance configured for deterministic classification.
    """
    settings = get_settings()
    model_kwargs = (
        {"reasoning_effort": settings.router_reasoning_effort}
        if settings.router_reasoning_effort
        else {}
    )
    return ChatGroq(
        model=settings.router_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=settings.router_max_tokens,
        timeout=settings.router_timeout_seconds,
        max_retries=0,
        model_kwargs=model_kwargs,
    )