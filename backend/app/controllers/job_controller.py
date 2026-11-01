import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

from app.controllers.chat_controller import answer_follow_up
from app.controllers.memory_controller import (
    acquire_conversation,
    ensure_conversation,
    load_context,
    load_job_response,
    memory_available,
    new_id,
    persist_outcome,
    record_job_start,
    release_conversation,
)
from app.controllers.pipeline_controller import run_research
from app.controllers.router_controller import build_reply, route_message
from app.core.errors import AppError
from app.core.runtime import EmitFn
from app.schemas.chat import ChatReply, Intent
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
from app.schemas.memory import ConversationContext
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


async def _emit(job: JobRecord, max_stream_chars: int, event: str, data: dict[str, Any]) -> None:
    """Progress callback for pipeline and chat code.

    Token chunks are also accumulated on the job, so the partial text can be saved
    if the job fails or is cancelled.
    """
    if event == "token" and len(job.stream_text) < max_stream_chars:
        job.stream_text += str(data.get("text", ""))
    await _publish(job, event, data)


async def _conclude(
    runtime: Runtime, job: JobRecord, status: JobStatus, event: str, data: dict[str, Any]
) -> None:
    """Saves the outcome to MongoDB, then publishes the terminal event.

    Saving first means that once the client sees the final event, the answer is
    already in the conversation history.
    """
    job.finished_at = time.time()
    await persist_outcome(runtime, job, status)
    await _publish(job, event, data, status=status)


async def _run_pipeline(
    runtime: Runtime, job: JobRecord, research_query: str, emit: EmitFn
) -> ResearchResponse:
    """Runs the research pipeline for a job inside the global concurrency limit.

    Args:
        runtime: Shared runtime.
        job: The job to execute.
        research_query: The self-contained research topic.
        emit: Progress callback.

    Returns:
        The research response.

    Raises:
        asyncio.TimeoutError: If the pipeline exceeds its time limit.
        AppError: If the pipeline fails.
    """
    if runtime.pipeline_semaphore.locked():
        await _publish(job, "queue_wait", {})
    async with runtime.pipeline_semaphore:
        await _publish(job, "research_started", {"query": research_query})
        return await asyncio.wait_for(
            run_research(runtime, research_query, emit),
            timeout=runtime.settings.pipeline_timeout_seconds,
        )


async def _load_job_context(runtime: Runtime, job: JobRecord) -> ConversationContext | None:
    """Loads conversation context for a job that belongs to a stored conversation."""
    if job.conversation_id and job.client_id and memory_available(runtime):
        return await load_context(runtime, job.conversation_id, job.client_id)
    return None


async def _run_job(runtime: Runtime, job: JobRecord) -> None:
    """Routes a message and replies directly, answers a follow-up or runs research.

    Greetings, off-topic and unsafe messages are answered immediately. Follow-ups
    are answered from the stored brief without any search. Only research requests
    wait for a pipeline slot. The outcome is saved from here, so it survives a
    browser disconnect.

    Args:
        runtime: Shared runtime.
        job: The job to execute.
    """
    emit = partial(_emit, job, runtime.settings.max_stream_chars)
    try:
        job.started_at = time.time()
        await _publish(
            job, "job_started", {"job_id": job.id, "query": job.query}, status=JobStatus.RUNNING
        )
        context = await _load_job_context(runtime, job)
        decision = await route_message(runtime, job.query, context)
        await _publish(job, "routed", {"intent": decision.intent.value})

        if decision.intent == Intent.RESEARCH:
            job.research_query = decision.standalone_query or job.query
            job.result = await _run_pipeline(runtime, job, job.research_query, emit)
            completed_data = {"job_id": job.id, "result": job.result.model_dump(mode="json")}
        elif decision.intent == Intent.FOLLOW_UP:
            text = await answer_follow_up(runtime, job.query, context, emit)
            job.reply = ChatReply(intent=Intent.FOLLOW_UP, text=text)
            completed_data = {"job_id": job.id, "reply": job.reply.model_dump(mode="json")}
        else:
            job.reply = build_reply(decision)
            completed_data = {"job_id": job.id, "reply": job.reply.model_dump(mode="json")}
    except asyncio.CancelledError:
        await _conclude(runtime, job, JobStatus.CANCELLED, "job_cancelled", {"job_id": job.id})
        raise
    except asyncio.TimeoutError:
        job.error = JobError(code="pipeline_timeout", message="Research timed out", status_code=504)
        logger.error("job.timeout job_id=%s", job.id)
        await _conclude(runtime, job, JobStatus.FAILED, "job_failed", job.error.model_dump())
    except AppError as exc:
        job.error = JobError(code=exc.code, message=exc.message, status_code=exc.status_code)
        logger.warning("job.failed job_id=%s code=%s message=%s", job.id, exc.code, exc.message)
        await _conclude(runtime, job, JobStatus.FAILED, "job_failed", job.error.model_dump())
    except Exception:
        logger.exception("job.unexpected_error job_id=%s", job.id)
        job.error = JobError(
            code="internal_error", message="Internal server error", status_code=500
        )
        await _conclude(runtime, job, JobStatus.FAILED, "job_failed", job.error.model_dump())
    else:
        logger.info("job.completed job_id=%s", job.id)
        await _conclude(runtime, job, JobStatus.COMPLETED, "job_completed", completed_data)


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


def submit_job(
    runtime: Runtime,
    query: str,
    *,
    job_id: str | None = None,
    client_id: str | None = None,
    conversation_id: str | None = None,
    user_message_id: str | None = None,
    request_id: str | None = None,
) -> JobRecord:
    """Registers a message as a job and starts it in the background.

    Must be called from a running event loop.

    Args:
        runtime: Shared runtime.
        query: The user's message.
        job_id: Optional pre-generated job id.
        client_id: The anonymous browser id that owns the job.
        conversation_id: The stored conversation the job belongs to.
        user_message_id: The id of the saved user message.
        request_id: The HTTP request id, for log correlation.

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
        id=job_id or new_id(),
        query=query,
        client_id=client_id,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        request_id=request_id,
        created_at=time.time(),
        condition=asyncio.Condition(),
    )
    job.events.append(
        JobEvent(seq=1, event="job_queued", data={"job_id": job.id}, timestamp=job.created_at)
    )
    runtime.jobs[job.id] = job
    job.task = asyncio.create_task(_run_job(runtime, job), name=f"research-job-{job.id}")
    logger.info(
        "job.submitted job_id=%s conversation_id=%s active=%d", job.id, conversation_id, active + 1
    )
    return job


async def submit_chat_job(
    runtime: Runtime,
    query: str,
    client_id: str | None,
    conversation_id: str | None,
    request_id: str | None,
) -> JobRecord:
    """Starts a job inside a stored conversation when memory is available.

    Saves the user's message, takes the conversation's single active-job lock and
    then starts the job. Without a client id or a database it falls back to a
    stateless job.

    Args:
        runtime: Shared runtime.
        query: The user's message.
        client_id: The anonymous browser id, or None.
        conversation_id: The conversation to continue, or None for a new one.
        request_id: The HTTP request id.

    Returns:
        The queued job record.

    Raises:
        AppError: With status 409 when the conversation already has a running
            job, or 429 when the service is busy.
    """
    stateless = partial(submit_job, runtime, query, client_id=client_id, request_id=request_id)
    if client_id is None or not memory_available(runtime):
        return stateless()

    conversation = await ensure_conversation(runtime, client_id, conversation_id, query)
    if conversation is None:
        return stateless()

    job_id = new_id()
    acquired = await acquire_conversation(runtime, conversation["_id"], client_id, job_id)
    if acquired is False:
        raise AppError(
            "Please wait for the current answer to finish.",
            status_code=409,
            code="conversation_busy",
        )
    if acquired is None:
        return stateless()

    try:
        user_message_id = await record_job_start(
            runtime, job_id, conversation["_id"], client_id, request_id, query
        )
        if user_message_id is None:
            await release_conversation(runtime, conversation["_id"], job_id)
            return stateless()
        return submit_job(
            runtime,
            query,
            job_id=job_id,
            client_id=client_id,
            conversation_id=conversation["_id"],
            user_message_id=user_message_id,
            request_id=request_id,
        )
    except BaseException:
        await release_conversation(runtime, conversation["_id"], job_id)
        raise


def _owns(job: JobRecord, client_id: str | None) -> bool:
    """Returns whether the caller may access the job.

    Jobs created without a client id are open to anyone who knows the job id.
    """
    return job.client_id is None or job.client_id == client_id


def get_job(runtime: Runtime, job_id: str, client_id: str | None = None) -> JobRecord:
    """Looks up an in-memory job by id for its owner.

    Args:
        runtime: Shared runtime.
        job_id: The job identifier.
        client_id: The anonymous browser id of the caller.

    Returns:
        The job record.

    Raises:
        AppError: With status 404 if the job does not exist, has expired or belongs
            to another client.
    """
    job = runtime.jobs.get(job_id)
    if job is None or not _owns(job, client_id):
        raise AppError("Job not found or expired", status_code=404, code="job_not_found")
    return job


async def get_job_view(runtime: Runtime, job_id: str, client_id: str | None) -> JobResponse:
    """Returns a job's public view, from memory or else from MongoDB.

    Args:
        runtime: Shared runtime.
        job_id: The job identifier.
        client_id: The anonymous browser id of the caller.

    Returns:
        The job view.

    Raises:
        AppError: With status 404 when the job is not found for this client.
    """
    job = runtime.jobs.get(job_id)
    if job is not None and _owns(job, client_id):
        return build_job_response(job)
    stored = await load_job_response(runtime, job_id, client_id) if client_id else None
    if stored is None:
        raise AppError("Job not found or expired", status_code=404, code="job_not_found")
    return stored


async def cancel_job(runtime: Runtime, job_id: str, client_id: str | None = None) -> JobRecord:
    """Cancels a queued or running job and waits until it has stopped.

    Cancelling a finished job is a no-op.

    Args:
        runtime: Shared runtime.
        job_id: The job identifier.
        client_id: The anonymous browser id of the caller.

    Returns:
        The job record in its final state.

    Raises:
        AppError: With status 404 if the job does not exist or has expired.
    """
    job = get_job(runtime, job_id, client_id)
    if job.task is not None and not job.task.done():
        job.task.cancel()
        await asyncio.gather(job.task, return_exceptions=True)
        logger.info("job.cancelled job_id=%s", job.id)
    return job


async def cancel_job_view(runtime: Runtime, job_id: str, client_id: str | None) -> JobResponse:
    """Cancels a job if it is still in memory, otherwise returns its stored state."""
    job = runtime.jobs.get(job_id)
    if job is not None and _owns(job, client_id):
        return build_job_response(await cancel_job(runtime, job_id, client_id))
    return await get_job_view(runtime, job_id, client_id)


async def cancel_conversation_jobs(runtime: Runtime, conversation_id: str) -> int:
    """Cancels every running job of a conversation, for example before deleting it.

    Args:
        runtime: Shared runtime.
        conversation_id: The conversation whose jobs should stop.

    Returns:
        The number of jobs that were cancelled.
    """
    tasks = [
        job.task
        for job in runtime.jobs.values()
        if job.conversation_id == conversation_id and job.task is not None and not job.task.done()
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return len(tasks)


async def _wait_until_finished(job: JobRecord) -> None:
    """Blocks until the job reaches a terminal status."""
    async with job.condition:
        await job.condition.wait_for(lambda: job.status in TERMINAL_STATUSES)


async def run_research_blocking(runtime: Runtime, query: str) -> ResearchResponse:
    """Runs a job and waits for its result, for clients that cannot poll or stream.

    Goes through the same router, queue and concurrency limits as background jobs,
    without a stored conversation. If the waiting request is cancelled, the
    underlying job is cancelled too.

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
    Last-Event-ID header misses nothing, including text already streamed. A
    keep-alive comment is sent while the job is quiet so proxies do not close the
    connection.

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


async def stream_stored_job(view: JobResponse) -> AsyncIterator[str]:
    """Yields a single terminal frame for a finished job that is only in MongoDB.

    Args:
        view: The stored job view.

    Yields:
        One SSE-formatted terminal event.
    """
    data: dict[str, Any] = {"job_id": view.job_id}
    if view.status == JobStatus.COMPLETED:
        event = "job_completed"
        if view.result is not None:
            data["result"] = view.result.model_dump(mode="json")
        elif view.reply is not None:
            data["reply"] = view.reply.model_dump(mode="json")
    elif view.status == JobStatus.CANCELLED:
        event = "job_cancelled"
    else:
        event = "job_failed"
        data = (
            view.error.model_dump()
            if view.error
            else {"code": "interrupted", "message": "The job did not finish.", "status_code": 500}
        )
    yield _format_sse(JobEvent(seq=1, event=event, data=data, timestamp=time.time()))


async def open_job_stream(
    runtime: Runtime, job_id: str, client_id: str | None, start_index: int
) -> AsyncIterator[str]:
    """Returns the SSE frame iterator for a job, live or from storage.

    Args:
        runtime: Shared runtime.
        job_id: The job identifier.
        client_id: The anonymous browser id of the caller.
        start_index: Number of events the client has already received.

    Returns:
        An async iterator of SSE frames.

    Raises:
        AppError: With status 404 when the job is not found for this client.
    """
    job = runtime.jobs.get(job_id)
    if job is not None and _owns(job, client_id):
        return stream_job_events(runtime, job, start_index)
    return stream_stored_job(await get_job_view(runtime, job_id, client_id))


def build_job_created(job: JobRecord, base_path: str) -> JobCreatedResponse:
    """Builds the 202 response for a newly submitted job.

    Args:
        job: The queued job.
        base_path: Path of the submit endpoint, used to derive the job URLs.

    Returns:
        The job id with its status, stream URL and conversation id.
    """
    status_url = f"{base_path.rstrip('/')}/{job.id}"
    return JobCreatedResponse(
        job_id=job.id,
        status=job.status,
        status_url=status_url,
        stream_url=f"{status_url}/stream",
        conversation_id=job.conversation_id,
        message_id=job.user_message_id,
    )


def build_job_response(job: JobRecord) -> JobResponse:
    """Builds the public view of a job, including its result, reply or error."""
    return JobResponse(
        job_id=job.id,
        status=job.status,
        query=job.query,
        conversation_id=job.conversation_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        events_count=len(job.events),
        result=job.result,
        reply=job.reply,
        error=job.error,
    )