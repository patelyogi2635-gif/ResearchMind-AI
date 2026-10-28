import logging
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
GITHUB_API_VERSION = "2022-11-28"
USER_AGENT = "researchmind-ai"
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
BACKOFF_INITIAL_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 5.0


def _is_retryable(exc: BaseException) -> bool:
    """Retries transport failures and transient HTTP statuses, never client errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


def _log_retry(retry_state: RetryCallState) -> None:
    """Logs a retry attempt before the backoff sleep."""
    error = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning("http.retry attempt=%d error=%s", retry_state.attempt_number, error)


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    max_attempts: int,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sends an HTTP request with exponential backoff and returns the JSON object.

    Args:
        client: Shared async HTTP client.
        method: HTTP method.
        url: Absolute request URL.
        headers: Request headers, including credentials.
        max_attempts: Total attempts before giving up.
        params: Optional query parameters.
        json_body: Optional JSON request body.

    Returns:
        The decoded JSON object, or an empty dict if the body is not an object.

    Raises:
        httpx.HTTPError: When all attempts fail or the error is not retryable.
        ValueError: When the response body is not valid JSON.
    """
    data: dict[str, Any] = {}
    retrying = AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=BACKOFF_INITIAL_SECONDS, max=BACKOFF_MAX_SECONDS),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            response = await client.request(
                method, url, headers=headers, params=params, json=json_body
            )
            response.raise_for_status()
            parsed = response.json()
            data = parsed if isinstance(parsed, dict) else {}
    return data


async def tavily_search(
    client: httpx.AsyncClient, query: str, max_results: int
) -> list[dict[str, Any]]:
    """Searches the web through the Tavily API.

    Args:
        client: Shared async HTTP client.
        query: Search query.
        max_results: Maximum number of results to return.

    Returns:
        Result dicts containing title, url, content and a relevance score.

    Raises:
        httpx.HTTPError: When the request fails after all retries.
        ValueError: When the response is not valid JSON.
    """
    settings = get_settings()
    data = await _request_json(
        client,
        "POST",
        TAVILY_SEARCH_URL,
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        json_body={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        },
        max_attempts=settings.http_max_attempts,
    )
    results = data.get("results", [])
    return results if isinstance(results, list) else []


async def github_search_repositories(
    client: httpx.AsyncClient, query: str, per_page: int
) -> list[dict[str, Any]]:
    """Searches GitHub repositories, sorted by stars.

    Args:
        client: Shared async HTTP client.
        query: GitHub search query, qualifiers allowed.
        per_page: Number of repositories to request.

    Returns:
        Repository dicts from the GitHub search API.

    Raises:
        httpx.HTTPError: When the request fails after all retries.
        ValueError: When the response is not valid JSON.
    """
    settings = get_settings()
    data = await _request_json(
        client,
        "GET",
        GITHUB_SEARCH_URL,
        headers={
            "Authorization": f"Bearer {settings.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": USER_AGENT,
        },
        params={"q": query, "sort": "stars", "order": "desc", "per_page": per_page},
        max_attempts=settings.http_max_attempts,
    )
    items = data.get("items", [])
    return items if isinstance(items, list) else []