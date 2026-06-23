"""
pipeline.py — orchestrates the full research pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: Why chain graphs instead of one big graph?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each graph is a self-contained unit:
  - Planner graph knows nothing about MCP tools
  - Research graph knows nothing about report writing
  - Writer graph knows nothing about query decomposition

This separation makes each graph:
  - Independently testable (unit test each graph alone)
  - Independently replaceable (swap the Planner without touching Writer)
  - Easier to debug (error traces point to a specific graph)

The pipeline service is the ONLY place that knows the full sequence.
It passes the state through each graph, accumulating results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: astream_events for Phase 3 SSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
graph.ainvoke() runs the graph and returns only the FINAL state.
graph.astream_events() yields events AS EACH NODE FIRES:
  - on_chain_start   → graph started
  - on_chain_end     → graph finished
  - on_chat_model_start → LLM call started
  - on_chat_model_stream → LLM token streaming

In Phase 3, we'll switch from ainvoke to astream_events so the
FastAPI SSE endpoint can push progress to the frontend in real time.
For now, ainvoke is simpler and returns the complete result.
"""

import logging
import time
from typing import AsyncGenerator

from app.state import ResearchState
from app.graphs.planner import planner_graph
from app.graphs.research import research_graph
from app.graphs.writer import writer_graph

logger = logging.getLogger(__name__)


async def run_research_pipeline(query: str) -> ResearchState:
    """
    Runs the full research pipeline for a given query.

    Flow:
      1. Planner graph  → adds sub_questions to state
      2. Research graph → adds search_results + sources to state
      3. Writer graph   → adds report to state

    Args:
        query: the user's research question

    Returns:
        Final ResearchState with all fields populated.
        Most important: state["report"] = the markdown report.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("[Pipeline] Starting research for: %s", query)

    # Initialize state — only query is set at the start
    state: ResearchState = {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "report": "",
    }

    # ── Step 1: Planner ──────────────────────────────────────────────
    logger.info("[Pipeline] Step 1/3 — Planner graph")
    t1 = time.time()

    state = await planner_graph.ainvoke(state)

    logger.info(
        "[Pipeline] Planner done in %.1fs → %d sub-questions",
        time.time() - t1, len(state.get("sub_questions", []))
    )

    # ── Step 2: Research ─────────────────────────────────────────────
    logger.info("[Pipeline] Step 2/3 — Research graph")
    t2 = time.time()

    state = await research_graph.ainvoke(state)

    logger.info(
        "[Pipeline] Research done in %.1fs → %d sources collected",
        time.time() - t2, len(state.get("sources", []))
    )

    # ── Step 3: Writer ───────────────────────────────────────────────
    logger.info("[Pipeline] Step 3/3 — Writer graph")
    t3 = time.time()

    state = await writer_graph.ainvoke(state)

    logger.info(
        "[Pipeline] Writer done in %.1fs → %d char report",
        time.time() - t3, len(state.get("report", ""))
    )

    total = time.time() - start_time
    logger.info("[Pipeline] Complete in %.1fs total", total)
    logger.info("=" * 60)

    return state


async def run_pipeline_streaming(query: str) -> AsyncGenerator[dict, None]:
    """
    Streaming version of the pipeline — yields progress events.
    Used in Phase 3 by the SSE endpoint.

    Yields dicts like:
      {"phase": "planning", "message": "Breaking down your query..."}
      {"phase": "planning_done", "data": {"sub_questions": [...]}}
      {"phase": "researching", "message": "Searching web and GitHub..."}
      {"phase": "research_done", "data": {"source_count": 8}}
      {"phase": "writing", "message": "Synthesizing report..."}
      {"phase": "done", "data": {"report": "# Report\n..."}}
      {"phase": "error", "message": "Something went wrong"}
    """
    state: ResearchState = {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "report": "",
    }

    try:
        # Phase 1
        yield {"phase": "planning", "message": "Breaking down your query into sub-questions..."}
        state = await planner_graph.ainvoke(state)
        yield {
            "phase": "planning_done",
            "message": f"Generated {len(state['sub_questions'])} sub-questions",
            "data": {"sub_questions": state["sub_questions"]},
        }

        # Phase 2
        yield {"phase": "researching", "message": "Searching web and GitHub repositories..."}
        state = await research_graph.ainvoke(state)
        yield {
            "phase": "research_done",
            "message": f"Found {len(state['sources'])} relevant sources",
            "data": {
                "source_count": len(state["sources"]),
                "sources": [
                    {"title": s["title"], "url": s["url"], "source": s["source"]}
                    for s in state["sources"]
                ],
            },
        }

        # Phase 3
        yield {"phase": "writing", "message": "Synthesizing research into a report..."}
        state = await writer_graph.ainvoke(state)
        yield {
            "phase": "done",
            "message": "Research complete",
            "data": {"report": state["report"]},
        }

    except Exception as e:
        logger.error("[Pipeline] Error during streaming pipeline: %s", e, exc_info=True)
        yield {"phase": "error", "message": str(e)}