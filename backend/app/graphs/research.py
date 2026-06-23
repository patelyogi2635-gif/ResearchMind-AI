"""
research.py — Graph 2: MCP-powered web + GitHub research.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: How LangGraph nodes call MCP tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MCP tools are LangChain BaseTool objects. You call them with:
  result = await tool.ainvoke({"query": "search term"})

The tool handles:
  - Sending the request to the MCP server subprocess
  - Waiting for the response
  - Returning a string or dict result

In this graph, we call tools directly (not via an LLM agent).
This gives us full control over what gets searched and how results
are processed — important for building reliable research pipelines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: Why loop outside the graph?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph supports dynamic loops via conditional edges.
But for Phase 2, we use a simpler pattern: the research node
itself loops over sub_questions. This is easier to debug and
still shows the key concept: state accumulation via operator.add.

In production you'd use LangGraph's map-reduce pattern for
true parallel execution. That's a Phase 4 enhancement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: operator.add and state accumulation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In state.py, search_results is defined as:
  Annotated[list[SearchResult], operator.add]

This means when a node returns:
  {"search_results": [new_item_1, new_item_2]}

LangGraph APPENDS to the existing list instead of replacing it.
So after each sub-question loop, search_results grows.
By the end, it contains results from ALL sub-questions.
"""

import json
import logging
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.state import ResearchState, SearchResult
from app.llm import get_fast_llm
from app.mcp_client import get_mcp_tools
from app.graphs.prompts import SEARCH_QUERY_SYSTEM, SEARCH_QUERY_USER
import re

logger = logging.getLogger(__name__)

# How many results to keep per sub-question per source
MAX_RESULTS_PER_SOURCE = 2
# How many total sources to pass to the Writer
MAX_TOTAL_SOURCES = 6


# ── Helper: parse Tavily results ──────────────────────────────────────

def _parse_tavily_results(raw: Any, sub_question: str) -> list[SearchResult]:
    """
    Tavily MCP returns formatted text, not JSON. Format is:
      Detailed Results:

      Title: Some Title
      URL: https://...
      Content: ...

      Title: Another Result
      ...
    """
    results = []
    try:
        # Unwrap any wrapper objects
        if hasattr(raw, 'content'):
            raw = raw.content
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        if hasattr(raw, 'text'):
            raw = raw.text
        if not isinstance(raw, str) or not raw.strip():
            return []

        # Split into blocks by double newline
        blocks = raw.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().splitlines()
            title = url = content = ""

            for line in lines:
                line = line.strip()
                if line.startswith("Title:"):
                    title = line[len("Title:"):].strip()
                elif line.startswith("URL:"):
                    url = line[len("URL:"):].strip()
                elif line.startswith("Content:"):
                    content = line[len("Content:"):].strip()

            # Only keep if we got at least a title and URL
            if title and url and url.startswith("http"):
                results.append(SearchResult(
                    sub_question=sub_question,
                    source="tavily",
                    title=title,
                    url=url,
                    content=content[:200],
                    relevance_score=0.7,
                ))

            if len(results) >= MAX_RESULTS_PER_SOURCE:
                break

    except Exception as e:
        logger.warning("[Research] Tavily parse error: %s", e)

    return results

# ── Helper: parse GitHub results ──────────────────────────────────────

def _parse_github_results(raw: Any, sub_question: str) -> list[SearchResult]:
    """
    GitHub MCP search_repositories may return text or JSON.
    Handle both formats.
    """
    results = []
    try:
        if hasattr(raw, 'content'):
            raw = raw.content
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        if hasattr(raw, 'text'):
            raw = raw.text
        if not isinstance(raw, str) or not raw.strip():
            return []

        # Try JSON first
        try:
            data = json.loads(raw)
            items = data.get("items", []) if isinstance(data, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                stars = item.get("stargazers_count", 0)
                if stars < 10:
                    continue
                desc = item.get("description") or ""
                if not desc:
                    continue
                score = min(1.0, (stars / 5000) ** 0.5)
                results.append(SearchResult(
                    sub_question=sub_question,
                    source="github",
                    title=item.get("full_name", ""),
                    url=item.get("html_url", ""),
                    content=f"{desc[:150]} | ⭐ {stars}",
                    relevance_score=score,
                ))
                if len(results) >= MAX_RESULTS_PER_SOURCE:
                    break
            return results
        except json.JSONDecodeError:
            pass

        # Fallback: parse text format same as Tavily
        blocks = raw.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().splitlines()
            title = url = content = ""
            for line in lines:
                line = line.strip()
                if line.startswith("Title:") or line.startswith("Name:"):
                    title = line.split(":", 1)[1].strip()
                elif line.startswith("URL:") or line.startswith("Link:"):
                    url = line.split(":", 1)[1].strip()
                    if url.startswith("//"):
                        url = "https:" + url
                elif line.startswith("Description:") or line.startswith("Content:"):
                    content = line.split(":", 1)[1].strip()
            if title and url and url.startswith("http"):
                results.append(SearchResult(
                    sub_question=sub_question,
                    source="github",
                    title=title,
                    url=url,
                    content=content[:150],
                    relevance_score=0.5,
                ))
            if len(results) >= MAX_RESULTS_PER_SOURCE:
                break

    except Exception as e:
        logger.warning("[Research] GitHub parse error: %s", e)
    return results

# ── Node: search web via Tavily MCP ───────────────────────────────────

async def search_web(state: ResearchState) -> dict:
    """
    Node: for each sub-question, calls Tavily MCP tool.
    Returns accumulated web search results.

    Why call LLM to optimize the search query?
    The sub-question might be "What are the limitations of LangGraph?"
    A better Tavily query is "LangGraph limitations drawbacks production".
    The fast 8b model does this cheaply and quickly.
    """
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        logger.warning("[Research] No sub-questions found in state")
        return {"search_results": []}

    # Get MCP tools
    tools = await get_mcp_tools(servers=["tavily"])
    tavily_tool = next((t for t in tools if "tavily" in t.name.lower()), None)

    if not tavily_tool:
        logger.error("[Research] Tavily tool not found")
        return {"search_results": []}

    llm = get_fast_llm()
    all_results: list[SearchResult] = []

    for sub_q in sub_questions:
        logger.info("[Research] Web searching for: %s", sub_q)

        # Step 1: optimize the search query
        try:
            msg = await llm.ainvoke([
                SystemMessage(content=SEARCH_QUERY_SYSTEM),
                HumanMessage(content=SEARCH_QUERY_USER.format(sub_question=sub_q)),
            ])
            search_query = msg.content.strip().strip('"')
        except Exception:
            search_query = sub_q  # fallback to raw sub-question

        # Step 2: call Tavily MCP
        try:
            raw = await tavily_tool.ainvoke({"query": search_query})
            logger.info("[Research] Tavily raw type: %s | preview: %s", type(raw).__name__, str(raw)[:300])
            results = _parse_tavily_results(raw, sub_q)
            all_results.extend(results)
            logger.info("[Research] Tavily returned %d results for: %s", len(results), search_query)
        except Exception as e:
            logger.error("[Research] Tavily call failed for '%s': %s", search_query, e)

    # operator.add will APPEND these to any existing search_results in state
    return {"search_results": all_results}


# ── Node: search GitHub repos via GitHub MCP ──────────────────────────

async def search_github(state: ResearchState) -> dict:
    """
    Node: for each sub-question, searches GitHub repositories.
    Returns accumulated GitHub results.

    GitHub results complement web results well:
    - Web: articles, docs, tutorials
    - GitHub: actual code, READMEs, real implementations
    """
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return {"search_results": []}

    tools = await get_mcp_tools(servers=["github"])
    github_tool = next(
        (t for t in tools if "search_repositories" in t.name.lower()),
        None
    )

    if not github_tool:
        logger.error("[Research] GitHub search_repositories tool not found")
        return {"search_results": []}

    all_results: list[SearchResult] = []

    for sub_q in sub_questions:
        logger.info("[Research] GitHub searching for: %s", sub_q)



        # Remove question words, keep nouns/terms
        stop_words = {"what", "how", "does", "is", "the", "are", "for", "and",
                      "of", "in", "to", "a", "an", "it", "its", "with", "vs",
                      "between", "compared", "differ", "difference", "using"}
        words = [w for w in re.sub(r'[^a-zA-Z0-9 ]', '', sub_q).lower().split()
                 if w not in stop_words]
        gh_query = " ".join(words[:4])
        if not gh_query:
            gh_query = sub_q.split()[0]

        try:
            raw = await github_tool.ainvoke({"query": gh_query})
            results = _parse_github_results(raw, sub_q)
            all_results.extend(results)
            logger.info("[Research] GitHub returned %d results for: %s", len(results), gh_query)
        except Exception as e:
            logger.error("[Research] GitHub call failed for '%s': %s", gh_query, e)

    return {"search_results": all_results}


# ── Node: score and deduplicate sources ───────────────────────────────

async def score_sources(state: ResearchState) -> dict:
    """
    Node: takes all raw search_results, deduplicates by URL,
    sorts by relevance_score, keeps top MAX_TOTAL_SOURCES.

    Why deduplicate?
    Different sub-questions often return the same URLs.
    Keeping duplicates wastes the Writer's context window.

    Returns: state.sources = clean, ranked, top-N list
    """
    raw_results = state.get("search_results", [])
    logger.info("[Research] Scoring %d raw results", len(raw_results))

    # Deduplicate by URL — keep highest score for each URL
    seen_urls: dict[str, SearchResult] = {}
    for result in raw_results:
        url = result.get("url", "")
        if not url:
            continue
        if url not in seen_urls or result["relevance_score"] > seen_urls[url]["relevance_score"]:
            seen_urls[url] = result

    # Sort by relevance score descending
    sorted_results = sorted(
        seen_urls.values(),
        key=lambda r: r["relevance_score"],
        reverse=True
    )

    # Keep top N
    top_sources = sorted_results[:MAX_TOTAL_SOURCES]

    logger.info(
        "[Research] After dedup+scoring: %d unique sources (kept top %d)",
        len(seen_urls), len(top_sources)
    )

    # Note: sources replaces (last-write-wins), not appends
    # That's correct — we want the final clean list here
    return {"sources": top_sources}


# ── Graph builder ─────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    """
    Builds and compiles the Research graph.

    Graph structure:
      START → search_web → search_github → score_sources → END

    search_web and search_github both write to search_results.
    Because of operator.add, their results ACCUMULATE in state.
    score_sources then reads the full accumulated list.

    Why sequential instead of parallel?
    Simpler to debug in Phase 2. In Phase 4 we'd use
    LangGraph's Send() API for true parallel fan-out.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("search_web", search_web)
    graph.add_node("search_github", search_github)
    graph.add_node("score_sources", score_sources)

    graph.set_entry_point("search_web")
    graph.add_edge("search_web", "search_github")
    graph.add_edge("search_github", "score_sources")
    graph.add_edge("score_sources", END)

    compiled = graph.compile()
    logger.info("[Research] Graph compiled successfully")
    return compiled


research_graph = build_research_graph()