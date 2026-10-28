import asyncio
import logging
import re
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from app.core.errors import AppError
from app.core.runtime import get_runtime
from app.core.search_api import github_search_repositories, tavily_search
from app.schemas.context import Runtime
from app.schemas.state import ResearchState, SearchResult

logger = logging.getLogger(__name__)

MAX_TOTAL_SOURCES = 6
MAX_GITHUB_RESULTS = 3
SNIPPET_CHARS = 700
MIN_GITHUB_STARS = 10
GITHUB_QUERY_WORDS = 4
GITHUB_MIN_STARS_FILTER = 50
GITHUB_STARS_SCORE_SCALE = 5000

GITHUB_STOP_WORDS = frozenset(
    {
        "what", "which", "how", "does", "do", "is", "the", "are", "for", "and", "of", "in", "to",
        "a", "an", "it", "its", "with", "vs", "versus", "between", "compared", "differ",
        "difference", "using", "should", "we", "use", "our", "my", "i", "can", "best", "good",
        "need", "want", "choose", "pick", "when", "why", "who", "where", "about", "on", "at",
        "by", "from", "be", "this", "that", "or",
    }
)


def _as_float(value: Any) -> float:
    """Converts a value to float, returning 0.0 when it is not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _tavily_to_results(
    items: list[dict[str, Any]], sub_question: str, min_score: float
) -> list[SearchResult]:
    """Maps Tavily results to search results using Tavily's own relevance score.

    Args:
        items: Raw Tavily result dicts.
        sub_question: The sub-question the results answer.
        min_score: Results scoring below this value are dropped.

    Returns:
        Search results with a title, an http URL and a score of at least min_score.
    """
    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        score = _as_float(item.get("score"))
        if not title or not url.startswith("http") or score < min_score:
            continue
        results.append(
            SearchResult(
                sub_question=sub_question,
                source="tavily",
                title=title,
                url=url,
                content=str(item.get("content") or "").strip()[:SNIPPET_CHARS],
                relevance_score=round(score, 3),
            )
        )
    return results


def _github_to_results(items: list[dict[str, Any]], query: str) -> list[SearchResult]:
    """Maps GitHub repositories to search results with health signals.

    Repositories with fewer than MIN_GITHUB_STARS stars or no description are
    dropped. The content carries stars, licence and last push date so the writer
    can judge maturity. The score grows with the square root of the star count.

    Args:
        items: Raw repository dicts from the GitHub search API.
        query: The research topic the repositories relate to.

    Returns:
        Search results ordered as received (already sorted by stars).
    """
    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        stars = item.get("stargazers_count")
        description = item.get("description") or ""
        url = item.get("html_url") or ""
        if not isinstance(stars, int) or stars < MIN_GITHUB_STARS or not description or not url:
            continue
        license_info = item.get("license")
        license_id = (
            license_info.get("spdx_id") if isinstance(license_info, dict) else None
        ) or "unspecified"
        last_push = str(item.get("pushed_at") or "")[:10] or "unknown"
        results.append(
            SearchResult(
                sub_question=query,
                source="github",
                title=item.get("full_name") or url,
                url=url,
                content=(
                    f"{description[:SNIPPET_CHARS]} | stars: {stars} | "
                    f"license: {license_id} | last push: {last_push}"
                ),
                relevance_score=min(1.0, (stars / GITHUB_STARS_SCORE_SCALE) ** 0.5),
            )
        )
    return results


def _build_github_query(query: str) -> str:
    """Reduces the research topic to a short GitHub repository search query.

    Drops stop words and numeric tokens, keeps the first few keywords and adds a
    minimum-stars qualifier so tiny or abandoned repositories are filtered out.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", query).lower().split()
    words = [w for w in cleaned if w not in GITHUB_STOP_WORDS and not w[0].isdigit()]
    keywords = " ".join(words[:GITHUB_QUERY_WORDS]) or query.split()[0]
    return f"{keywords} stars:>{GITHUB_MIN_STARS_FILTER}"


async def _search_web_one(runtime: Runtime, sub_question: str) -> list[SearchResult]:
    """Searches Tavily for one sub-question, returning an empty list on failure."""
    settings = runtime.settings
    try:
        async with runtime.search_semaphore:
            items = await tavily_search(
                runtime.http, sub_question, settings.tavily_results_per_query
            )
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("search_web.call_failed question=%r error=%s", sub_question, exc)
        return []
    results = _tavily_to_results(items, sub_question, settings.min_relevance_score)
    logger.info("search_web.done question=%r results=%d", sub_question, len(results))
    return results


async def search_web(state: ResearchState, config: RunnableConfig) -> dict:
    """Graph node that searches the web for all sub-questions in parallel.

    Concurrency across all users is bounded by the runtime's search semaphore. A
    failed sub-question does not fail the node.

    Args:
        state: Current pipeline state containing `sub_questions`.
        config: Run config carrying the shared runtime.

    Returns:
        A partial state update with `search_results`.
    """
    runtime = get_runtime(config)
    sub_questions = state.get("sub_questions", [])
    groups = await asyncio.gather(*(_search_web_one(runtime, q) for q in sub_questions))
    collected = [result for group in groups for result in group]
    logger.info("search_web.complete results=%d", len(collected))
    return {"search_results": collected}


async def search_github(state: ResearchState, config: RunnableConfig) -> dict:
    """Graph node that searches GitHub repositories for the main research topic.

    Runs in parallel with planning and web search. Uses one query built from the
    topic, since per-sub-question queries are too generic to match repositories.

    Args:
        state: Current pipeline state containing the user's `query`.
        config: Run config carrying the shared runtime.

    Returns:
        A partial state update with `search_results`.
    """
    runtime = get_runtime(config)
    query = state["query"]
    github_query = _build_github_query(query)
    try:
        async with runtime.search_semaphore:
            items = await github_search_repositories(
                runtime.http, github_query, runtime.settings.github_results_per_query
            )
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("search_github.call_failed query=%r error=%s", github_query, exc)
        return {"search_results": []}

    results = _github_to_results(items, query)[:MAX_GITHUB_RESULTS]
    logger.info(
        "search_github.done query=%r fetched=%d kept=%d", github_query, len(items), len(results)
    )
    return {"search_results": results}


async def score_sources(state: ResearchState, config: RunnableConfig) -> dict:
    """Graph node that deduplicates results by URL and keeps the top sources.

    Args:
        state: Current pipeline state containing accumulated `search_results`.
        config: Run config (unused, kept for a uniform node signature).

    Returns:
        A partial state update with `sources`, ranked by relevance score.

    Raises:
        AppError: If no usable source was retrieved.
    """
    raw_results = state.get("search_results", [])
    unique: dict[str, SearchResult] = {}
    for result in raw_results:
        url = result.get("url", "")
        if not url:
            continue
        if url not in unique or result["relevance_score"] > unique[url]["relevance_score"]:
            unique[url] = result

    ranked = sorted(unique.values(), key=lambda r: r["relevance_score"], reverse=True)
    top_sources = ranked[:MAX_TOTAL_SOURCES]
    logger.info(
        "search.scored raw=%d unique=%d kept=%d", len(raw_results), len(unique), len(top_sources)
    )
    if not top_sources:
        raise AppError(
            "No sources could be retrieved for this topic. Try rephrasing the query.",
            status_code=502,
            code="no_sources_found",
        )
    return {"sources": top_sources}