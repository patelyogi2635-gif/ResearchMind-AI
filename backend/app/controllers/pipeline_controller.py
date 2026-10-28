import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.controllers.planner_controller import decompose_query
from app.controllers.search_controller import score_sources, search_github, search_web
from app.controllers.writer_controller import select_sources, synthesize_brief
from app.core.errors import AppError
from app.schemas.context import Runtime
from app.schemas.research import ResearchResponse, SourceResponse
from app.schemas.state import ResearchState

logger = logging.getLogger(__name__)

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


async def _noop_emit(event: str, data: dict[str, Any]) -> None:
    """Default emitter used when no progress listener is attached."""
    return None


def _get_emit(config: RunnableConfig) -> EmitFn:
    """Returns the progress emitter from the run config, or a no-op."""
    return (config.get("configurable") or {}).get("emit") or _noop_emit


def _summarize_update(update: dict | None) -> dict[str, Any]:
    """Builds a small, client-safe summary of what a node produced."""
    if not update:
        return {}
    summary: dict[str, Any] = {}
    if "sub_questions" in update:
        summary["sub_questions"] = update["sub_questions"]
    if "search_results" in update:
        summary["results_count"] = len(update["search_results"])
    if "sources" in update:
        summary["sources_count"] = len(update["sources"])
    if "report" in update:
        summary["report_chars"] = len(update["report"])
    return summary


def track_progress(name: str, node: Callable[[ResearchState, RunnableConfig], Awaitable[dict]]):
    """Wraps a graph node so it emits start and completion events with timing.

    Args:
        name: Node name used in events and logs.
        node: The async node function taking (state, config).

    Returns:
        An async node with the same behaviour that reports its progress.
    """

    async def wrapped(state: ResearchState, config: RunnableConfig) -> dict:
        emit = _get_emit(config)
        await emit("node_started", {"node": name})
        started = time.perf_counter()
        update = await node(state, config)
        duration = round(time.perf_counter() - started, 2)
        logger.info("pipeline.node_done node=%s duration=%.2fs", name, duration)
        await emit(
            "node_completed",
            {"node": name, "duration_seconds": duration, **_summarize_update(update)},
        )
        return update

    return wrapped


def build_pipeline_graph():
    """Builds the single research graph.

    Structure: planner and GitHub search start in parallel, web search follows the
    planner, and scoring waits for both searches.

    START -> decompose_query -> search_web ----\\
    START -> search_github --------------------+-> score_sources -> select_sources
                                                   -> synthesize_brief -> END
    """
    graph = StateGraph(ResearchState)
    graph.add_node("decompose_query", track_progress("decompose_query", decompose_query))
    graph.add_node("search_web", track_progress("search_web", search_web))
    graph.add_node("search_github", track_progress("search_github", search_github))
    graph.add_node("score_sources", track_progress("score_sources", score_sources))
    graph.add_node("select_sources", track_progress("select_sources", select_sources))
    graph.add_node("synthesize_brief", track_progress("synthesize_brief", synthesize_brief))

    graph.add_edge(START, "decompose_query")
    graph.add_edge(START, "search_github")
    graph.add_edge("decompose_query", "search_web")
    graph.add_edge(["search_web", "search_github"], "score_sources")
    graph.add_edge("score_sources", "select_sources")
    graph.add_edge("select_sources", "synthesize_brief")
    graph.add_edge("synthesize_brief", END)
    return graph.compile()


pipeline_graph = build_pipeline_graph()


async def run_research(runtime: Runtime, query: str, emit: EmitFn) -> ResearchResponse:
    """Runs the full R&D pipeline and builds the API response.

    Args:
        runtime: Shared runtime with the HTTP client and concurrency limits.
        query: The technology topic to evaluate.
        emit: Async callback receiving (event, data) progress updates.

    Returns:
        A ResearchResponse with the structured brief, markdown report and sources.

    Raises:
        AppError: If research finds no sources, the brief cannot be generated, or
            the pipeline fails unexpectedly.
    """
    started = time.perf_counter()
    logger.info("research.start query=%r", query)
    state: ResearchState = {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "brief": None,
        "report": "",
    }
    config: RunnableConfig = {"configurable": {"runtime": runtime, "emit": emit}}
    try:
        final_state = await pipeline_graph.ainvoke(state, config=config)
    except AppError:
        raise
    except Exception as exc:
        raise AppError("Research pipeline failed", status_code=502, code="pipeline_failed") from exc

    brief = final_state["brief"]
    if brief is None or not final_state["report"]:
        raise AppError("Pipeline finished without a brief", status_code=500, code="empty_brief")

    duration = time.perf_counter() - started
    sources = final_state["sources"]
    logger.info("research.done duration=%.2fs sources=%d", duration, len(sources))
    return ResearchResponse(
        query=final_state["query"],
        sub_questions=final_state["sub_questions"],
        sources=[
            SourceResponse(
                id=index,
                title=source["title"],
                url=source["url"],
                source=source["source"],
                relevance_score=source["relevance_score"],
            )
            for index, source in enumerate(sources, 1)
        ],
        brief=brief,
        report=final_state["report"],
        source_count=len(sources),
        duration_seconds=round(duration, 2),
    )