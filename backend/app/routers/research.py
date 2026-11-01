from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.controllers.job_controller import (
    build_job_created,
    build_job_response,
    cancel_job_view,
    get_job_view,
    open_job_stream,
    parse_last_event_id,
    run_research_blocking,
    submit_chat_job,
)
from app.core.runtime import get_app_runtime, get_client_id
from app.schemas.common import ErrorResponse
from app.schemas.context import Runtime
from app.schemas.jobs import JobCreatedResponse, JobResponse
from app.schemas.memory import ChatRequest
from app.schemas.research import ResearchRequest, ResearchResponse

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
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

    Stateless: no conversation is stored. Uses the same queue and concurrency
    limits as background jobs. Prefer the job endpoints for chat, streaming and
    follow-up questions.
    """
    return await run_research_blocking(runtime, payload.query)


@router.post(
    "/research/jobs",
    response_model=JobCreatedResponse,
    status_code=202,
    responses=ERROR_RESPONSES,
)
async def create_research_job(
    payload: ChatRequest,
    request: Request,
    runtime: Runtime = Depends(get_app_runtime),
    client_id: str | None = Depends(get_client_id),
) -> JobCreatedResponse:
    """Starts a message as a background job and returns its ids immediately.

    With an X-Client-Id header the message is saved in a conversation, which is
    created when no conversation_id is sent. The response carries the
    conversation_id to use for follow-up questions.
    """
    job = await submit_chat_job(
        runtime, payload.query, client_id, payload.conversation_id, request.state.request_id
    )
    return build_job_created(job, request.url.path)


@router.get("/research/jobs/{job_id}", response_model=JobResponse, responses=ERROR_RESPONSES)
async def get_research_job(
    job_id: str,
    runtime: Runtime = Depends(get_app_runtime),
    client_id: str | None = Depends(get_client_id),
) -> JobResponse:
    """Returns a job's status and its result, reply or error once it has finished.

    Falls back to MongoDB for jobs that expired from memory or survived a restart.
    """
    return await get_job_view(runtime, job_id, client_id)


@router.get("/research/jobs/{job_id}/stream", responses=ERROR_RESPONSES)
async def stream_research_job(
    job_id: str,
    request: Request,
    runtime: Runtime = Depends(get_app_runtime),
    client_id: str | None = Depends(get_client_id),
) -> StreamingResponse:
    """Streams job progress and answer text as Server-Sent Events.

    Supports reconnecting with the Last-Event-ID header to replay missed events.
    Closing the stream does not cancel the job. EventSource clients pass the
    client_id query parameter because they cannot send headers.
    """
    start_index = parse_last_event_id(request.headers.get("last-event-id"))
    frames = await open_job_stream(runtime, job_id, client_id, start_index)
    return StreamingResponse(frames, media_type="text/event-stream", headers=SSE_HEADERS)


@router.delete("/research/jobs/{job_id}", response_model=JobResponse, responses=ERROR_RESPONSES)
async def cancel_research_job(
    job_id: str,
    runtime: Runtime = Depends(get_app_runtime),
    client_id: str | None = Depends(get_client_id),
) -> JobResponse:
    """Cancels a queued or running job. Cancelling a finished job changes nothing."""
    return await cancel_job_view(runtime, job_id, client_id)