import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.llm import get_fast_llm
from app.core.mcp_client import get_mcp_tools
from app.core.prompts import SEARCH_QUERY_SYSTEM, SEARCH_QUERY_USER
from app.schemas.state import ResearchState, SearchResult

logger = logging.getLogger(__name__)

MAX_RESULTS_PER_SOURCE = 2
MAX_TOTAL_SOURCES = 6
SNIPPET_CHARS = 500
TAVILY_DEFAULT_SCORE = 0.7
GITHUB_TEXT_SCORE = 0.5
MIN_GITHUB_STARS = 10
GITHUB_QUERY_WORDS = 4

FIELD_LABELS = {
    "title": "title",
    "name": "title",
    "url": "url",
    "link": "url",
    "content": "content",
    "description": "content",
}

GITHUB_STOP_WORDS = frozenset(
    {
        "what", "how", "does", "is", "the", "are", "for", "and", "of", "in", "to", "a", "an",
        "it", "its", "with", "vs", "between", "compared", "differ", "difference", "using",
    }
)


def _unwrap_text(raw: Any) -> str:
    """Extracts plain text from the different shapes MCP tools return."""
    if hasattr(raw, "content"):
        raw = raw.content
    if isinstance(raw, list):
        parts = [_unwrap_text(item) for item in raw]
        return "\n\n".join(part for part in parts if part)
    if isinstance(raw, dict):
        return str(raw.get("text", "")).strip()
    if hasattr(raw, "text"):
        raw = raw.text
    return raw.strip() if isinstance(raw, str) else ""


def _parse_text_blocks(text: str, source: str, sub_question: str, score: float) -> list[SearchResult]:
    """Parses 'Title/URL/Content' style text blocks into search results.

    Args:
        text: Raw tool output split into blocks by blank lines.
        source: Source label stored on each result ("tavily" or "github").
        sub_question: The sub-question the results answer.
        score: Relevance score assigned to every parsed result.

    Returns:
        Up to MAX_RESULTS_PER_SOURCE results that have a title and an http URL.
    """
    results: list[SearchResult] = []
    for block in text.split("\n\n"):
        fields = {"title": "", "url": "", "content": ""}
        current: str | None = None
        for line in block.strip().splitlines():
            label, separator, value = line.partition(":")
            key = FIELD_LABELS.get(label.strip().lower()) if separator else None
            if key:
                fields[key] = value.strip()
                current = key
            elif current == "content":
                fields["content"] += " " + line.strip()

        url = fields["url"]
        if url.startswith("//"):
            url = "https:" + url
        if fields["title"] and url.startswith("http"):
            results.append(
                SearchResult(
                    sub_question=sub_question,
                    source=source,
                    title=fields["title"],
                    url=url,
                    content=fields["content"].strip()[:SNIPPET_CHARS],
                    relevance_score=score,
                )
            )
        if len(results) >= MAX_RESULTS_PER_SOURCE:
            break
    return results


def _parse_tavily_results(raw: Any, sub_question: str) -> list[SearchResult]:
    """Parses Tavily MCP output into search results."""
    text = _unwrap_text(raw)
    if not text:
        return []
    return _parse_text_blocks(text, "tavily", sub_question, TAVILY_DEFAULT_SCORE)


def _parse_github_results(raw: Any, sub_question: str) -> list[SearchResult]:
    """Parses GitHub MCP output (JSON or text) into search results.

    Repositories with fewer than MIN_GITHUB_STARS stars or no description are
    dropped. The score grows with the square root of the star count.
    """
    text = _unwrap_text(raw)
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _parse_text_blocks(text, "github", sub_question, GITHUB_TEXT_SCORE)

    items = data.get("items", []) if isinstance(data, dict) else []
    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        stars = item.get("stargazers_count", 0)
        description = item.get("description") or ""
        if stars < MIN_GITHUB_STARS or not description:
            continue
        results.append(
            SearchResult(
                sub_question=sub_question,
                source="github",
                title=item.get("full_name", ""),
                url=item.get("html_url", ""),
                content=f"{description[:SNIPPET_CHARS]} | stars: {stars}",
                relevance_score=min(1.0, (stars / 5000) ** 0.5),
            )
        )
        if len(results) >= MAX_RESULTS_PER_SOURCE:
            break
    return results


def _build_github_query(sub_question: str) -> str:
    """Reduces a natural-language sub-question to a short GitHub search query."""
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", sub_question).lower().split()
    words = [word for word in cleaned if word not in GITHUB_STOP_WORDS]
    return " ".join(words[:GITHUB_QUERY_WORDS]) or sub_question.split()[0]


def _find_tool(tools: list, *keywords: str):
    """Returns the first tool whose name contains every keyword, or None."""
    for tool in tools:
        name = tool.name.lower()
        if all(keyword in name for keyword in keywords):
            return tool
    return None


async def _load_tool(server: str, *keywords: str):
    """Loads one MCP tool, returning None instead of raising when unavailable."""
    try:
        tools = await get_mcp_tools(servers=[server])
    except Exception as exc:
        logger.error("search.mcp_unavailable server=%s error=%s", server, exc)
        return None
    tool = _find_tool(tools, *keywords)
    if tool is None:
        logger.error("search.tool_missing server=%s keywords=%s", server, keywords)
    return tool


async def _optimize_query(llm, sub_question: str) -> str:
    """Rewrites a sub-question into a search-engine friendly query."""
    try:
        message = await llm.ainvoke(
            [
                SystemMessage(content=SEARCH_QUERY_SYSTEM),
                HumanMessage(content=SEARCH_QUERY_USER.format(sub_question=sub_question)),
            ]
        )
    except Exception as exc:
        logger.warning("search_web.query_optimize_failed error=%s", exc)
        return sub_question
    return str(message.content).strip().strip('"') or sub_question


async def search_web(state: ResearchState) -> dict:
    """Graph node that searches the web via the Tavily MCP server.

    Args:
        state: Current pipeline state containing `sub_questions`.

    Returns:
        A partial state update with `search_results`. The list is empty when the
        tool is unavailable or every call failed.
    """
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        logger.warning("search_web.skipped reason=no_sub_questions")
        return {"search_results": []}

    tool = await _load_tool("tavily", "search")
    if tool is None:
        return {"search_results": []}

    llm = get_fast_llm()
    collected: list[SearchResult] = []
    for sub_question in sub_questions:
        search_query = await _optimize_query(llm, sub_question)
        try:
            raw = await tool.ainvoke({"query": search_query})
        except Exception as exc:
            logger.error("search_web.call_failed query=%r error=%s", search_query, exc)
            continue
        results = _parse_tavily_results(raw, sub_question)
        logger.info("search_web.done query=%r results=%d", search_query, len(results))
        collected.extend(results)
    return {"search_results": collected}


async def search_github(state: ResearchState) -> dict:
    """Graph node that searches GitHub repositories via the GitHub MCP server.

    Args:
        state: Current pipeline state containing `sub_questions`.

    Returns:
        A partial state update with `search_results`.
    """
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return {"search_results": []}

    tool = await _load_tool("github", "search_repositories")
    if tool is None:
        return {"search_results": []}

    collected: list[SearchResult] = []
    for sub_question in sub_questions:
        github_query = _build_github_query(sub_question)
        try:
            raw = await tool.ainvoke({"query": github_query})
        except Exception as exc:
            logger.error("search_github.call_failed query=%r error=%s", github_query, exc)
            continue
        results = _parse_github_results(raw, sub_question)
        logger.info("search_github.done query=%r results=%d", github_query, len(results))
        collected.extend(results)
    return {"search_results": collected}


async def score_sources(state: ResearchState) -> dict:
    """Graph node that deduplicates results by URL and keeps the top sources.

    Args:
        state: Current pipeline state containing accumulated `search_results`.

    Returns:
        A partial state update with `sources`, ranked by relevance score.
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
    return {"sources": top_sources}


def build_research_graph():
    """Builds and compiles the research graph.

    Structure: search_web -> search_github -> score_sources -> END
    """
    graph = StateGraph(ResearchState)
    graph.add_node("search_web", search_web)
    graph.add_node("search_github", search_github)
    graph.add_node("score_sources", score_sources)
    graph.set_entry_point("search_web")
    graph.add_edge("search_web", "search_github")
    graph.add_edge("search_github", "score_sources")
    graph.add_edge("score_sources", END)
    return graph.compile()


research_graph = build_research_graph()
