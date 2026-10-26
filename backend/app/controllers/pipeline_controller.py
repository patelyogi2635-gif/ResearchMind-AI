import logging
import time

from app.controllers.planner_controller import planner_graph
from app.controllers.search_controller import research_graph
from app.controllers.writer_controller import writer_graph
from app.core.errors import AppError
from app.schemas.research import ResearchResponse, SourceResponse
from app.schemas.state import ResearchState

logger = logging.getLogger(__name__)

PIPELINE_STAGES = (
    ("planner", planner_graph),
    ("research", research_graph),
    ("writer", writer_graph),
)


async def run_research_pipeline(query: str) -> ResearchState:
    """Runs the planner, research and writer graphs in sequence.

    Args:
        query: The technology topic to evaluate.

    Returns:
        The final pipeline state with `brief` and `report` populated.

    Raises:
        AppError: If research finds no sources or the brief cannot be generated.
    """
    state: ResearchState = {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "brief": None,
        "report": "",
    }
    for name, graph in PIPELINE_STAGES:
        started = time.perf_counter()
        logger.info("pipeline.stage_start stage=%s", name)
        state = await graph.ainvoke(state)
        logger.info(
            "pipeline.stage_done stage=%s duration=%.2fs", name, time.perf_counter() - started
        )
        if name == "research" and not state["sources"]:
            raise AppError(
                "No sources could be retrieved for this topic. Try rephrasing the query.",
                status_code=502,
                code="no_sources_found",
            )
    return state


async def run_research(query: str) -> ResearchResponse:
    """Runs the full R&D pipeline and builds the API response.

    Args:
        query: The technology topic to evaluate.

    Returns:
        A ResearchResponse with the structured brief, markdown report and sources.

    Raises:
        AppError: If the pipeline fails for a known or unexpected reason.
    """
    started = time.perf_counter()
    logger.info("research.start query=%r", query)
    try:
        state = await run_research_pipeline(query)
    except AppError:
        raise
    except Exception as exc:
        raise AppError("Research pipeline failed", status_code=502, code="pipeline_failed") from exc

    brief = state["brief"]
    if brief is None or not state["report"]:
        raise AppError("Pipeline finished without a brief", status_code=500, code="empty_brief")

    duration = time.perf_counter() - started
    logger.info("research.done duration=%.2fs sources=%d", duration, len(state["sources"]))
    return ResearchResponse(
        query=state["query"],
        sub_questions=state["sub_questions"],
        sources=[
            SourceResponse(
                id=index,
                title=source["title"],
                url=source["url"],
                source=source["source"],
                relevance_score=source["relevance_score"],
            )
            for index, source in enumerate(state["sources"], 1)
        ],
        brief=brief,
        report=state["report"],
        source_count=len(state["sources"]),
        duration_seconds=round(duration, 2),
    )
