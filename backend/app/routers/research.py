from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.controllers.job_controller import (
    build_job_created,
    build_job_response,
    cancel_job,
    get_job,
    parse_last_event_id,
    run_research_blocking,
    stream_job_events,
    submit_job,
)
from app.core.runtime import get_app_runtime
from app.schemas.common import ErrorResponse
from app.schemas.context import Runtime
from app.schemas.jobs import JobCreatedResponse, JobResponse
from app.schemas.research import ResearchRequest, ResearchResponse

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

ERROR_RESPONSES = {
    404: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


@router.post("/research", response_model=ResearchResponse, responses=ERROR_RESPONSES)
async def create_research(
    payload: ResearchRequest, runtime: Runtime = Depends(get_app_runtime)
) -> ResearchResponse:
    """Generates an R&D evaluation brief and waits for the result.

    Uses the same queue and concurrency limits as background jobs. Prefer the job
    endpoints for long-running or streamed requests.
    """
    return await run_research_blocking(runtime, payload.query)


@router.post(
    "/research/jobs",
    response_model=JobCreatedResponse,
    status_code=202,
    responses=ERROR_RESPONSES,
)
async def create_research_job(
    payload: ResearchRequest,
    request: Request,
    runtime: Runtime = Depends(get_app_runtime),
) -> JobCreatedResponse:
    """Starts a research job in the background and returns its id immediately."""
    job = submit_job(runtime, payload.query)
    return build_job_created(job, request.url.path)


@router.get("/research/jobs/{job_id}", response_model=JobResponse, responses=ERROR_RESPONSES)
async def get_research_job(
    job_id: str, runtime: Runtime = Depends(get_app_runtime)
) -> JobResponse:
    """Returns a job's status, and its result or error once it has finished."""
    return build_job_response(get_job(runtime, job_id))


@router.get("/research/jobs/{job_id}/stream", responses=ERROR_RESPONSES)
async def stream_research_job(
    job_id: str, request: Request, runtime: Runtime = Depends(get_app_runtime)
) -> StreamingResponse:
    """Streams job progress as Server-Sent Events.

    Supports reconnecting with the Last-Event-ID header to replay missed events.
    Closing the stream does not cancel the job.
    """
    job = get_job(runtime, job_id)
    start_index = parse_last_event_id(request.headers.get("last-event-id"))
    return StreamingResponse(
        stream_job_events(runtime, job, start_index),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.delete("/research/jobs/{job_id}", response_model=JobResponse, responses=ERROR_RESPONSES)
async def cancel_research_job(
    job_id: str, runtime: Runtime = Depends(get_app_runtime)
) -> JobResponse:
    """Cancels a queued or running job. Cancelling a finished job changes nothing."""
    return build_job_response(await cancel_job(runtime, job_id))