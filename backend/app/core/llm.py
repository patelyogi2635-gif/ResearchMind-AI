import re
from functools import lru_cache

from langchain_groq import ChatGroq

from app.core.config import get_settings

RATE_LIMIT_STATUS = 429
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 2.0
MAX_RATE_LIMIT_WAIT_SECONDS = 6.0
RETRY_IN_PATTERN = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b", re.IGNORECASE)


def rate_limit_delay(exc: BaseException) -> float | None:
    """Returns how long to wait before retrying after a provider rate limit.

    Reads the Retry-After header when present, otherwise parses the "try again in"
    hint in the error message, otherwise falls back to a short default.

    Args:
        exc: The exception raised by the model call.

    Returns:
        Seconds to wait, or None when the error is not a rate limit.
    """
    if getattr(exc, "status_code", None) != RATE_LIMIT_STATUS:
        return None
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    match = RETRY_IN_PATTERN.search(str(exc))
    if match:
        value = float(match.group(1))
        return value / 1000 if match.group(2).lower() == "ms" else value
    return DEFAULT_RATE_LIMIT_WAIT_SECONDS


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


@lru_cache
def get_chat_llm() -> ChatGroq:
    """Returns the cached model used for follow-up answers and streamed summaries.

    Retries are left to the stream helper, which only retries before any token has
    been shown to the user.

    Returns:
        A ChatGroq instance configured for conversational answers.
    """
    settings = get_settings()
    model_kwargs = (
        {"reasoning_effort": settings.chat_reasoning_effort}
        if settings.chat_reasoning_effort
        else {}
    )
    return ChatGroq(
        model=settings.chat_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
        max_tokens=settings.chat_max_tokens,
        timeout=settings.chat_timeout_seconds,
        max_retries=0,
        model_kwargs=model_kwargs,
    )