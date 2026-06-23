"""
writer.py — Graph 3: Report synthesis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: Context window management
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Groq llama-3.3-70b has a 32k token context window.
Each source snippet is ~100 tokens. With 10 sources that's ~1000 tokens
for source content, leaving ~31k for the prompt and response.

But we still format sources carefully:
  - Number them so LLM can cite "[1]", "[2]" etc.
  - Include title, URL, and content snippet
  - Keep snippets short (500 chars from research.py)

This gives the LLM enough context to write accurate citations
without overwhelming it with irrelevant text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: Why a separate select_sources node?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Separating "select" from "write" is good practice:
  - select_sources: pure data transformation (no LLM)
  - synthesize_report: pure LLM call

Each node has one job. Easy to test each in isolation.
If the report quality is bad, you know which node to fix.
"""

import logging
from langgraph.graph import StateGraph, END

from app.state import ResearchState, SearchResult
from app.llm import get_llm
from app.graphs.prompts import WRITER_SYSTEM, WRITER_USER
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Max sources to include in the writer prompt
MAX_WRITER_SOURCES = 4


# ── Helper: format sources as readable text ───────────────────────────

def _format_sources_for_prompt(sources: list[SearchResult]) -> str:
    """
    Converts SearchResult list into numbered text the LLM can read.

    Format:
      [1] Title (tavily)
      URL: https://...
      Content: ...snippet...

      [2] ...

    Numbered format lets LLM write citations like [1], [2] naturally.
    """
    if not sources:
        return "No sources available."

    lines = []
    for i, source in enumerate(sources, 1):
        lines.append(f"[{i}] {source['title']} ({source['source']})")
        lines.append(f"URL: {source['url']}")
        lines.append(f"Content: {source['content']}")
        lines.append("")  # blank line between sources

    return "\n".join(lines)


# ── Node: select best sources for writer ─────────────────────────────

async def select_sources(state: ResearchState) -> dict:
    all_sources = state.get("sources", [])

    if not all_sources:
        logger.warning("[Writer] No sources in state — report will be thin")
        # Must return a valid state key — never return {}
        return {"sources": []}

    web_sources = [s for s in all_sources if s["source"] == "tavily"]
    gh_sources = [s for s in all_sources if s["source"] == "github"]

    selected = web_sources[:3] + gh_sources[:1]
    selected.sort(key=lambda s: s["relevance_score"], reverse=True)
    selected = selected[:MAX_WRITER_SOURCES]

    logger.info("[Writer] Selected %d sources", len(selected))
    return {"sources": selected}


# ── Node: synthesize report using Groq LLM ───────────────────────────

async def synthesize_report(state: ResearchState) -> dict:
    """
    Node: the final LLM call that writes the research report.

    Takes state.query + state.sources, returns state.report.

    The report is markdown-formatted with:
      - Executive summary
      - Structured sections
      - Inline citations [Title](URL)
      - Sources list at end
    """
    query = state["query"]
    sources = state.get("sources", [])

    logger.info("[Writer] Synthesizing report for: %s", query)
    logger.info("[Writer] Using %d sources", len(sources))

    # Format sources for the prompt
    sources_text = _format_sources_for_prompt(sources)

    # Use higher temperature for writing (more natural prose)
    llm = get_llm(temperature=0.4)

    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=WRITER_USER.format(
            query=query,
            sources_text=sources_text,
        )),
    ]

    response = await llm.ainvoke(messages)
    report = response.content.strip()

    logger.info("[Writer] Report generated: %d characters", len(report))

    return {"report": report}


# ── Graph builder ─────────────────────────────────────────────────────

def build_writer_graph() -> StateGraph:
    """
    Builds and compiles the Writer graph.

    Graph structure:
      START → select_sources → synthesize_report → END

    Simple two-node pipeline.
    select_sources is pure Python (no LLM cost).
    synthesize_report is the one expensive LLM call.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("select_sources", select_sources)
    graph.add_node("synthesize_report", synthesize_report)

    graph.set_entry_point("select_sources")
    graph.add_edge("select_sources", "synthesize_report")
    graph.add_edge("synthesize_report", END)

    compiled = graph.compile()
    logger.info("[Writer] Graph compiled successfully")
    return compiled


writer_graph = build_writer_graph()