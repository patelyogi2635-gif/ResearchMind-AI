from fastapi import APIRouter

from app.controllers.pipeline_controller import run_research
from app.schemas.common import ErrorResponse
from app.schemas.research import ResearchRequest, ResearchResponse

router = APIRouter()


@router.post(
    "/research",
    response_model=ResearchResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def create_research(request: ResearchRequest) -> ResearchResponse:
    """Generates an R&D evaluation brief for a technology topic.

    Runs planner, research and writer graphs and returns a structured brief, its
    markdown rendering and the cited sources. The call blocks for the duration of
    the pipeline.
    """
    return await run_research(request.query)
