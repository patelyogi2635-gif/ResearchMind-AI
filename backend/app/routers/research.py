"""
routers/research.py — HTTP endpoints for the research pipeline.

Endpoints:
  POST /api/research       → runs full pipeline, returns complete report (sync)
  POST /api/research/quick → runs pipeline, returns just the report text

Phase 3 will add:
  GET  /api/research/stream/{session_id} → SSE streaming version
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.pipeline import run_research_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The research question to investigate",
        example="How does LangGraph compare to CrewAI for building production AI agents?",
    )


class SourceResponse(BaseModel):
    title: str
    url: str
    source: str       # "tavily" or "github"
    relevance_score: float


class ResearchResponse(BaseModel):
    query: str
    sub_questions: list[str]
    sources: list[SourceResponse]
    report: str
    source_count: int


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """
    Runs the full research pipeline:
      Planner → Research → Writer

    Returns the complete report + metadata.
    This is a blocking call — takes 20-60 seconds depending on query.

    Phase 3 will add a streaming version that pushes progress via SSE.
    """
    logger.info("[Router] Research request: %s", request.query)

    try:
        state = await run_research_pipeline(request.query)
    except Exception as e:
        logger.error("[Router] Pipeline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Research pipeline failed: {str(e)}")

    if not state.get("report"):
        raise HTTPException(status_code=500, detail="Pipeline completed but report is empty")

    return ResearchResponse(
        query=state["query"],
        sub_questions=state.get("sub_questions", []),
        sources=[
            SourceResponse(
                title=s["title"],
                url=s["url"],
                source=s["source"],
                relevance_score=s["relevance_score"],
            )
            for s in state.get("sources", [])
        ],
        report=state["report"],
        source_count=len(state.get("sources", [])),
    )


@router.get("/research/health")
async def research_health():
    """Quick check that the research router is registered."""
    return {
        "status": "ok",
        "graphs": ["planner", "research", "writer"],
        "pipeline": "ready",
    }