import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

from app.controllers.pipeline_controller import run_research
from app.controllers.router_controller import build_reply, route_message
from app.core.errors import AppError
from app.schemas.chat import Intent
from app.schemas.context import Runtime
from app.schemas.jobs import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    JobCreatedResponse,
    JobError,
    JobEvent,
    JobRecord,
    JobResponse,
    JobStatus,
)
from app.schemas.research import ResearchResponse

logger = logging.getLogger(__name__)


async def _publish(
    job: JobRecord, event: str, data: dict[str, Any], status: JobStatus | None = None
) -> None:
    """Appends an event to the job and wakes every stream waiting on it.

    The status change and the event are applied under the same lock, so a stream
    never observes a terminal status without also seeing the final event.

    Args:
        job: The job to publish to.
        event: Event name such as "node_completed".
        data: JSON-serializable event payload.
        status: Optional new job status applied together with the event.
    """
    async with job.condition:
        if status is not None:
            job.status = status
        job.events.append(
            JobEvent(seq=len(job.events) + 1, event=event, data=data, timestamp=time.time())
        )
        job.condition.notify_all()


async def _finish(job: JobRecord, status: JobStatus, event: str, data: dict[str, Any]) -> None:
    """Marks a job as finished and publishes its terminal event."""
    job.finished_at = time.time()
    await _publish(job, event, data, status=status)


async def _run_pipeline(runtime: Runtime, job: JobRecord) -> ResearchResponse:
    """Runs the research pipeline for a job inside the global concurrency limit.

    Args:
        runtime: Shared runtime.
        job: The job to execute.

    Returns:
        The research response.

    Raises:
        asyncio.TimeoutError: If the pipeline exceeds its time limit.
        AppError: If the pipeline fails.
    """
    if runtime.pipeline_semaphore.locked():
        await _publish(job, "queue_wait", {})
    async with runtime.pipeline_semaphore:
        await _publish(job, "research_started", {"query": job.query})
        return await asyncio.wait_for(
            run_research(runtime, job.query, partial(_publish, job)),
            timeout=runtime.settings.pipeline_timeout_seconds,
        )


async def _run_job(runtime: Runtime, job: JobRecord) -> None:
    """Routes a message and either replies directly or runs the research pipeline.

    Greetings, off-topic and unsafe messages are answered immediately without
    touching the pipeline. Only research requests wait for a pipeline slot.

    Args:
        runtime: Shared runtime.
        job: The job to execute.
    """
    try:
        job.started_at = time.time()
        await _publish(
            job, "job_started", {"job_id": job.id, "query": job.query}, status=JobStatus.RUNNING
        )
        decision = await route_message(runtime, job.query)
        await _publish(job, "routed", {"intent": decision.intent.value})

        if decision.intent == Intent.RESEARCH:
            job.result = await _run_pipeline(runtime, job)
            completed_data = {"job_id": job.id, "result": job.result.model_dump(mode="json")}
        else:
            job.reply = build_reply(decision)
            completed_data = {"job_id": job.id, "reply": job.reply.model_dump(mode="json")}
    except asyncio.CancelledError:
        await _finish(job, JobStatus.CANCELLED, "job_cancelled", {"job_id": job.id})
        raise
    except asyncio.TimeoutError:
        job.error = JobError(code="pipeline_timeout", message="Research timed out", status_code=504)
        logger.error("job.timeout job_id=%s", job.id)
        await _finish(job, JobStatus.FAILED, "job_failed", job.error.model_dump())
    except AppError as exc:
        job.error = JobError(code=exc.code, message=exc.message, status_code=exc.status_code)
        logger.warning("job.failed job_id=%s code=%s message=%s", job.id, exc.code, exc.message)
        await _finish(job, JobStatus.FAILED, "job_failed", job.error.model_dump())
    except Exception:
        logger.exception("job.unexpected_error job_id=%s", job.id)
        job.error = JobError(
            code="internal_error", message="Internal server error", status_code=500
        )
        await _finish(job, JobStatus.FAILED, "job_failed", job.error.model_dump())
    else:
        logger.info("job.completed job_id=%s", job.id)
        await _finish(job, JobStatus.COMPLETED, "job_completed", completed_data)


def purge_expired_jobs(runtime: Runtime) -> int:
    """Removes finished jobs past their TTL and trims the store to its size limit.

    Args:
        runtime: Shared runtime holding the job registry.

    Returns:
        The number of jobs removed because their TTL expired.
    """
    settings = runtime.settings
    now = time.time()
    expired = [
        job_id
        for job_id, job in runtime.jobs.items()
        if job.finished_at is not None and now - job.finished_at > settings.job_ttl_seconds
    ]
    for job_id in expired:
        del runtime.jobs[job_id]

    overflow = len(runtime.jobs) - settings.max_stored_jobs
    if overflow > 0:
        finished = sorted(
            (job for job in runtime.jobs.values() if job.finished_at is not None),
            key=lambda job: job.finished_at,
        )
        for job in finished[:overflow]:
            del runtime.jobs[job.id]
    return len(expired)


def submit_job(runtime: Runtime, query: str) -> JobRecord:
    """Registers a message as a job and starts it in the background.

    Must be called from a running event loop.

    Args:
        runtime: Shared runtime.
        query: The user's message.

    Returns:
        The queued job record.

    Raises:
        AppError: With status 429 when too many jobs are already queued or running.
    """
    purge_expired_jobs(runtime)
    active = sum(1 for job in runtime.jobs.values() if job.status in ACTIVE_STATUSES)
    if active >= runtime.settings.max_pending_jobs:
        raise AppError(
            "The service is busy. Please retry shortly.", status_code=429, code="server_busy"
        )

    job = JobRecord(
        id=uuid.uuid4().hex,
        query=query,
        created_at=time.time(),
        condition=asyncio.Condition(),
    )
    job.events.append(
        JobEvent(seq=1, event="job_queued", data={"job_id": job.id}, timestamp=job.created_at)
    )
    runtime.jobs[job.id] = job
    job.task = asyncio.create_task(_run_job(runtime, job), name=f"research-job-{job.id}")
    logger.info("job.submitted job_id=%s active=%d", job.id, active + 1)
    return job


def get_job(runtime: Runtime, job_id: str) -> JobRecord:
    """Looks up a job by id.

    Args:
        runtime: Shared runtime.
        job_id: The job identifier.

    Returns:
        The job record.

    Raises:
        AppError: With status 404 if the job does not exist or has expired.
    """
    job = runtime.jobs.get(job_id)
    if job is None:
        raise AppError("Job not found or expired", status_code=404, code="job_not_found")
    return job


async def cancel_job(runtime: Runtime, job_id: str) -> JobRecord:
    """Cancels a queued or running job and waits until it has stopped.

    Cancelling a finished job is a no-op.

    Args:
        runtime: Shared runtime.
        job_id: The job identifier.

    Returns:
        The job record in its final state.

    Raises:
        AppError: With status 404 if the job does not exist or has expired.
    """
    job = get_job(runtime, job_id)
    if job.task is not None and not job.task.done():
        job.task.cancel()
        await asyncio.gather(job.task, return_exceptions=True)
        logger.info("job.cancelled job_id=%s", job.id)
    return job


async def _wait_until_finished(job: JobRecord) -> None:
    """Blocks until the job reaches a terminal status."""
    async with job.condition:
        await job.condition.wait_for(lambda: job.status in TERMINAL_STATUSES)


async def run_research_blocking(runtime: Runtime, query: str) -> ResearchResponse:
    """Runs a job and waits for its result, for clients that cannot poll or stream.

    Goes through the same router, queue and concurrency limits as background jobs.
    If the waiting request is cancelled, the underlying job is cancelled too.

    Args:
        runtime: Shared runtime.
        query: The user's message.

    Returns:
        The completed research response.

    Raises:
        AppError: If the job is rejected, fails, times out or is cancelled, or with
            status 422 when the message was not a research request.
    """
    job = submit_job(runtime, query)
    try:
        await _wait_until_finished(job)
    except asyncio.CancelledError:
        if job.task is not None:
            job.task.cancel()
        raise

    if job.status == JobStatus.COMPLETED:
        if job.result is not None:
            return job.result
        if job.reply is not None:
            raise AppError(job.reply.text, status_code=422, code="not_a_research_request")
    if job.error is not None:
        raise AppError(job.error.message, status_code=job.error.status_code, code=job.error.code)
    raise AppError("Research job was cancelled", status_code=409, code="job_cancelled")


def _format_sse(event: JobEvent) -> str:
    """Formats a job event as a Server-Sent Events frame."""
    payload = json.dumps(event.data, separators=(",", ":"), ensure_ascii=False)
    return f"id: {event.seq}\nevent: {event.event}\ndata: {payload}\n\n"


def parse_last_event_id(value: str | None) -> int:
    """Converts a Last-Event-ID header into the index to resume streaming from.

    Args:
        value: Raw header value, or None.

    Returns:
        The number of events already received, never negative.
    """
    try:
        return max(int(value), 0) if value is not None else 0
    except ValueError:
        return 0


async def stream_job_events(
    runtime: Runtime, job: JobRecord, start_index: int = 0
) -> AsyncIterator[str]:
    """Yields a job's events as SSE frames until the job reaches a terminal state.

    Replays events from `start_index`, so a reconnecting client with a
    Last-Event-ID header misses nothing. A keep-alive comment is sent while the
    job is quiet so proxies do not close the connection.

    Args:
        runtime: Shared runtime.
        job: The job to stream.
        start_index: Number of events the client has already received.

    Yields:
        SSE-formatted strings.
    """
    index = start_index
    keepalive = runtime.settings.sse_keepalive_seconds
    while True:
        try:
            async with job.condition:
                await asyncio.wait_for(
                    job.condition.wait_for(
                        lambda: len(job.events) > index or job.status in TERMINAL_STATUSES
                    ),
                    timeout=keepalive,
                )
                batch = job.events[index:]
        except asyncio.TimeoutError:
            yield ": keep-alive\n\n"
            continue

        for event in batch:
            yield _format_sse(event)
        index += len(batch)
        if job.status in TERMINAL_STATUSES and index >= len(job.events):
            return


def build_job_created(job: JobRecord, base_path: str) -> JobCreatedResponse:
    """Builds the 202 response for a newly submitted job.

    Args:
        job: The queued job.
        base_path: Path of the submit endpoint, used to derive the job URLs.

    Returns:
        The job id with its status and stream URLs.
    """
    status_url = f"{base_path.rstrip('/')}/{job.id}"
    return JobCreatedResponse(
        job_id=job.id,
        status=job.status,
        status_url=status_url,
        stream_url=f"{status_url}/stream",
    )


def build_job_response(job: JobRecord) -> JobResponse:
    """Builds the public view of a job, including its result, reply or error."""
    return JobResponse(
        job_id=job.id,
        status=job.status,
        query=job.query,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        events_count=len(job.events),
        result=job.result,
        reply=job.reply,
        error=job.error,
    )