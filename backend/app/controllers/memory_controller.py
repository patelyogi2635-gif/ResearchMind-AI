import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.core.errors import AppError
from app.core.mongo import expiry_time
from app.schemas.context import Runtime
from app.schemas.jobs import JobError, JobRecord, JobResponse, JobStatus
from app.schemas.memory import (
    ConversationContext,
    ConversationListResponse,
    ConversationResponse,
    ConversationSummary,
    LookupResponse,
    MessageResponse,
)
from app.schemas.research import ResearchResponse
from app.schemas.chat import ChatReply, Intent

logger = logging.getLogger(__name__)

TITLE_CHARS = 60
TURN_CHARS = 300
MAX_TEXT_CHARS = 20000
BRIEF_CONTEXT_CHARS = 1800
OPTION_SUMMARY_CHARS = 120
RATIONALE_CHARS = 240
MAX_CONTEXT_LIMITATIONS = 2
WRITE_RETRY_DELAY_SECONDS = 0.2
INTERRUPTED_MESSAGE = "The server restarted before this finished. Please ask again."


def new_id() -> str:
    """Returns a new unguessable 128-bit identifier as 32 hex characters."""
    return uuid.uuid4().hex


def memory_available(runtime: Runtime) -> bool:
    """Returns whether MongoDB is connected and memory features can be used."""
    return runtime.mongo_db is not None


def _require_memory(runtime: Runtime) -> None:
    """Raises a 503 error when conversation history cannot be served."""
    if not memory_available(runtime):
        raise AppError(
            "Conversation history is unavailable right now.",
            status_code=503,
            code="memory_unavailable",
        )


def _timeout(runtime: Runtime) -> float:
    """Returns the per-operation database timeout in seconds."""
    return runtime.settings.mongodb_timeout_ms / 1000


async def _read(runtime: Runtime, label: str, operation: Awaitable, default: Any = None) -> Any:
    """Awaits a database read with a timeout, returning a default on failure."""
    try:
        return await asyncio.wait_for(operation, timeout=_timeout(runtime))
    except (PyMongoError, asyncio.TimeoutError) as exc:
        logger.error("memory.read_failed op=%s error=%s", label, str(exc)[:200])
        return default


async def _write(runtime: Runtime, label: str, make_operation: Callable[[], Awaitable]) -> bool:
    """Runs a database write with a timeout and a few quick retries.

    A duplicate key error counts as success, which makes saving idempotent.

    Args:
        runtime: Shared runtime.
        label: Short operation name for logs.
        make_operation: Factory returning a fresh awaitable for every attempt.

    Returns:
        True when the write succeeded, False after all attempts failed.
    """
    attempts = runtime.settings.mongodb_write_attempts
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.wait_for(make_operation(), timeout=_timeout(runtime))
            return True
        except DuplicateKeyError:
            return True
        except (PyMongoError, asyncio.TimeoutError) as exc:
            logger.warning(
                "memory.write_retry op=%s attempt=%d error=%s", label, attempt, str(exc)[:200]
            )
            if attempt < attempts:
                await asyncio.sleep(WRITE_RETRY_DELAY_SECONDS * attempt)
    logger.error("memory.write_failed op=%s", label)
    return False


def _title(text: str) -> str:
    """Builds a short conversation title from the first message."""
    return " ".join(text.split())[:TITLE_CHARS] or "New conversation"


async def ensure_conversation(
    runtime: Runtime, client_id: str, conversation_id: str | None, first_message: str
) -> dict | None:
    """Returns the client's conversation, creating a new one when needed.

    An unknown id, or an id owned by another client, starts a new conversation.

    Args:
        runtime: Shared runtime.
        client_id: The anonymous browser id of the caller.
        conversation_id: The conversation to continue, or None.
        first_message: The message used to title a new conversation.

    Returns:
        The conversation document, or None when the database is unavailable.
    """
    db = runtime.mongo_db
    if conversation_id:
        existing = await _read(
            runtime,
            "conversation.get",
            db.conversations.find_one({"_id": conversation_id, "client_id": client_id}),
        )
        if existing:
            return existing

    now = time.time()
    document = {
        "_id": new_id(),
        "client_id": client_id,
        "title": _title(first_message),
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
        "active_job_id": None,
        "expires_at": expiry_time(runtime.settings),
    }
    created = await _write(runtime, "conversation.create", lambda: db.conversations.insert_one(document))
    return document if created else None


async def acquire_conversation(
    runtime: Runtime, conversation_id: str, client_id: str, job_id: str
) -> bool | None:
    """Atomically marks a job as the only active one in a conversation.

    Args:
        runtime: Shared runtime.
        conversation_id: The conversation to lock.
        client_id: The owner of the conversation.
        job_id: The job taking the lock.

    Returns:
        True when acquired, False when another job holds it, None when the database
        could not be reached.
    """
    db = runtime.mongo_db
    result = await _read(
        runtime,
        "conversation.acquire",
        db.conversations.update_one(
            {"_id": conversation_id, "client_id": client_id, "active_job_id": None},
            {"$set": {"active_job_id": job_id, "updated_at": time.time()}},
        ),
    )
    if result is None:
        return None
    return result.matched_count == 1


async def release_conversation(runtime: Runtime, conversation_id: str, job_id: str) -> None:
    """Releases the conversation lock if it is still held by the given job."""
    db = runtime.mongo_db
    await _write(
        runtime,
        "conversation.release",
        lambda: db.conversations.update_one(
            {"_id": conversation_id, "active_job_id": job_id}, {"$set": {"active_job_id": None}}
        ),
    )


async def record_job_start(
    runtime: Runtime,
    job_id: str,
    conversation_id: str,
    client_id: str,
    request_id: str | None,
    query: str,
) -> str | None:
    """Saves the user's message and a queued job document.

    Args:
        runtime: Shared runtime.
        job_id: The job identifier.
        conversation_id: The conversation the message belongs to.
        client_id: The owner of the conversation.
        request_id: The HTTP request id, for log correlation.
        query: The user's message text.

    Returns:
        The new user message id, or None when saving failed.
    """
    db = runtime.mongo_db
    now = time.time()
    expires = expiry_time(runtime.settings)
    message_id = new_id()
    message = {
        "_id": message_id,
        "conversation_id": conversation_id,
        "client_id": client_id,
        "role": "user",
        "kind": "user",
        "text": query[:MAX_TEXT_CHARS],
        "job_id": job_id,
        "request_id": request_id,
        "status": "completed",
        "created_at": now,
        "expires_at": expires,
    }
    job = {
        "_id": job_id,
        "conversation_id": conversation_id,
        "client_id": client_id,
        "request_id": request_id,
        "user_message_id": message_id,
        "assistant_message_id": None,
        "query": query[:MAX_TEXT_CHARS],
        "status": JobStatus.QUEUED.value,
        "created_at": now,
        "finished_at": None,
        "error": None,
        "expires_at": expires,
    }
    saved = await _write(runtime, "message.user", lambda: db.messages.insert_one(message))
    saved = saved and await _write(runtime, "job.create", lambda: db.jobs.insert_one(job))
    return message_id if saved else None


def _assistant_document(job: JobRecord, status: JobStatus, now: float, runtime: Runtime) -> dict:
    """Builds the assistant message document for a finished job."""
    result = job.result.model_dump(mode="json") if job.result is not None else None
    if job.result is not None:
        kind, intent = "research", Intent.RESEARCH.value
        text = job.result.brief.executive_summary if job.result.brief else ""
    elif job.reply is not None:
        kind, intent, text = "reply", job.reply.intent.value, job.reply.text
    elif status == JobStatus.CANCELLED:
        kind, intent, text = "cancelled", None, job.stream_text
    else:
        kind, intent = "error", None
        text = job.stream_text or (job.error.message if job.error else "Something went wrong.")
    return {
        "_id": new_id(),
        "conversation_id": job.conversation_id,
        "client_id": job.client_id,
        "role": "assistant",
        "kind": kind,
        "text": text[:MAX_TEXT_CHARS],
        "intent": intent,
        "job_id": job.id,
        "request_id": job.request_id,
        "status": status.value,
        "result": result,
        "error": job.error.model_dump() if job.error else None,
        "created_at": now,
        "expires_at": expiry_time(runtime.settings),
    }


async def persist_outcome(runtime: Runtime, job: JobRecord, status: JobStatus) -> None:
    """Saves a finished job's answer once, then releases the conversation lock.

    Called from the background job, so the answer is saved even if the browser
    disconnected. It never raises: failures are retried and then logged.

    Args:
        runtime: Shared runtime.
        job: The finished job.
        status: The final job status.
    """
    if not memory_available(runtime) or not job.conversation_id or not job.client_id:
        return
    db = runtime.mongo_db
    now = time.time()
    message = _assistant_document(job, status, now, runtime)
    await _write(runtime, "message.assistant", lambda: db.messages.insert_one(message))
    await _write(
        runtime,
        "job.finish",
        lambda: db.jobs.update_one(
            {"_id": job.id},
            {
                "$set": {
                    "status": status.value,
                    "finished_at": now,
                    "assistant_message_id": message["_id"],
                    "error": message["error"],
                }
            },
        ),
    )
    await _write(
        runtime,
        "conversation.finish",
        lambda: db.conversations.update_one(
            {"_id": job.conversation_id, "active_job_id": job.id},
            {"$set": {"active_job_id": None, "updated_at": now}, "$inc": {"message_count": 2}},
        ),
    )


async def recover_interrupted_jobs(runtime: Runtime) -> int:
    """Marks jobs left running by a previous process as interrupted.

    Also saves an explanatory assistant message for each and clears stale
    conversation locks. Assumes a single server instance.

    Args:
        runtime: Shared runtime.

    Returns:
        The number of jobs that were marked interrupted.
    """
    if not memory_available(runtime):
        return 0
    db = runtime.mongo_db
    stale = await _read(
        runtime,
        "recover.find",
        _fetch(db.jobs, {"status": {"$in": [JobStatus.QUEUED.value, JobStatus.RUNNING.value]}}, "created_at", 1, 500),
        default=[],
    )
    now = time.time()
    error = JobError(code="interrupted", message=INTERRUPTED_MESSAGE, status_code=500)
    for job in stale:
        message = {
            "_id": new_id(),
            "conversation_id": job["conversation_id"],
            "client_id": job["client_id"],
            "role": "assistant",
            "kind": "error",
            "text": INTERRUPTED_MESSAGE,
            "intent": None,
            "job_id": job["_id"],
            "request_id": job.get("request_id"),
            "status": JobStatus.INTERRUPTED.value,
            "result": None,
            "error": error.model_dump(),
            "created_at": now,
            "expires_at": expiry_time(runtime.settings),
        }
        await _write(runtime, "recover.message", lambda m=message: db.messages.insert_one(m))
        await _write(
            runtime,
            "recover.job",
            lambda j=job, m=message: db.jobs.update_one(
                {"_id": j["_id"]},
                {"$set": {"status": JobStatus.INTERRUPTED.value, "finished_at": now, "assistant_message_id": m["_id"], "error": error.model_dump()}},
            ),
        )
    await _write(
        runtime,
        "recover.locks",
        lambda: db.conversations.update_many(
            {"active_job_id": {"$ne": None}}, {"$set": {"active_job_id": None}}
        ),
    )
    if stale:
        logger.warning("startup.recovery interrupted_jobs=%d", len(stale))
    return len(stale)


async def _fetch(collection, query: dict, sort_field: str, direction: int, limit: int) -> list[dict]:
    """Returns up to `limit` documents sorted by one field."""
    cursor = collection.find(query).sort(sort_field, direction).limit(limit)
    return await cursor.to_list(length=limit)


def build_brief_context(result: dict) -> str:
    """Builds a compact text summary of a stored research result for prompts.

    Keeps the topic, options, recommendation, key limitations and numbered
    sources, so follow-up prompts stay small.

    Args:
        result: A stored ResearchResponse as a dict.

    Returns:
        A summary capped at BRIEF_CONTEXT_CHARS characters.
    """
    brief = result.get("brief") or {}
    recommendation = brief.get("recommendation") or {}
    lines = [
        f"Topic: {result.get('query', '')}",
        f"Title: {brief.get('title', '')}",
        f"Summary: {brief.get('executive_summary', '')}",
        "Options:",
    ]
    for option in brief.get("options") or []:
        summary = str(option.get("summary", ""))[:OPTION_SUMMARY_CHARS]
        lines.append(f"- {option.get('name', '')} ({option.get('maturity', 'unknown')}): {summary}")
    lines.append(
        f"Recommendation: {recommendation.get('choice', '')} "
        f"({recommendation.get('confidence', '')} confidence). "
        f"{str(recommendation.get('rationale', ''))[:RATIONALE_CHARS]}"
    )
    for limitation in (brief.get("limitations") or [])[:MAX_CONTEXT_LIMITATIONS]:
        lines.append(f"Limitation: {limitation}")
    lines.append("Sources:")
    for source in result.get("sources") or []:
        lines.append(f"[{source.get('id')}] {source.get('title', '')}")
    return "\n".join(lines)[:BRIEF_CONTEXT_CHARS]


def _render_turn(document: dict, blocked_jobs: set[str]) -> str | None:
    """Renders one stored message as a short context line, or None to skip it."""
    text = (document.get("text") or "").strip()
    if document["role"] == "user":
        return None if document.get("job_id") in blocked_jobs else f"User: {text[:TURN_CHARS]}"
    kind = document.get("kind")
    if kind == "research":
        brief = (document.get("result") or {}).get("brief") or {}
        choice = (brief.get("recommendation") or {}).get("choice", "")
        return f"Assistant: [research brief] {brief.get('title', '')}, recommended: {choice}"
    if kind == "reply" and document.get("intent") != Intent.UNSAFE.value:
        return f"Assistant: {text[:TURN_CHARS]}"
    return None


async def load_context(
    runtime: Runtime, conversation_id: str, client_id: str
) -> ConversationContext | None:
    """Loads recent turns and a summary of the latest brief for routing and answers.

    Messages that were refused as unsafe, and the requests that caused them, are
    left out so they cannot influence later turns.

    Args:
        runtime: Shared runtime.
        conversation_id: The conversation to read.
        client_id: The owner of the conversation.

    Returns:
        The context, or None when memory is unavailable or the read failed.
    """
    if not memory_available(runtime):
        return None
    db = runtime.mongo_db
    limit = runtime.settings.history_turns * 2
    scope = {"conversation_id": conversation_id, "client_id": client_id}
    recent = await _read(runtime, "context.recent", _fetch(db.messages, scope, "created_at", -1, limit))
    if recent is None:
        return None
    latest = await _read(
        runtime,
        "context.brief",
        db.messages.find_one({**scope, "kind": "research"}, sort=[("created_at", -1)]),
    )
    recent.reverse()
    blocked_jobs = {
        doc["job_id"]
        for doc in recent
        if doc["role"] == "assistant" and doc.get("intent") == Intent.UNSAFE.value
    }
    turns = [line for doc in recent if (line := _render_turn(doc, blocked_jobs))]
    brief = build_brief_context(latest["result"]) if latest and latest.get("result") else None
    return ConversationContext(turns=turns, brief_summary=brief)


def _summary(document: dict) -> ConversationSummary:
    """Maps a conversation document to its API summary."""
    return ConversationSummary(
        conversation_id=document["_id"],
        title=document.get("title", ""),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
        message_count=document.get("message_count", 0),
        active_job_id=document.get("active_job_id"),
    )


def _message_response(document: dict) -> MessageResponse:
    """Maps a message document to its API representation."""
    result = None
    if document.get("result"):
        try:
            result = ResearchResponse.model_validate(document["result"])
        except ValidationError:
            logger.warning("memory.result_invalid message_id=%s", document["_id"])
    error = JobError.model_validate(document["error"]) if document.get("error") else None
    return MessageResponse(
        message_id=document["_id"],
        conversation_id=document["conversation_id"],
        role=document["role"],
        kind=document.get("kind", ""),
        text=document.get("text", ""),
        intent=document.get("intent"),
        job_id=document.get("job_id"),
        request_id=document.get("request_id"),
        status=document.get("status"),
        created_at=document["created_at"],
        result=result,
        error=error,
    )


async def list_conversations(
    runtime: Runtime, client_id: str, limit: int, before: float | None
) -> ConversationListResponse:
    """Lists a client's conversations, newest first, with cursor pagination.

    Args:
        runtime: Shared runtime.
        client_id: The anonymous browser id.
        limit: Page size.
        before: Only conversations updated before this time, for the next page.

    Returns:
        The page of conversations and the cursor for the next page.

    Raises:
        AppError: With status 503 when the database is unavailable.
    """
    _require_memory(runtime)
    query: dict[str, Any] = {"client_id": client_id}
    if before is not None:
        query["updated_at"] = {"$lt": before}
    documents = await _read(
        runtime, "conversations.list", _fetch(runtime.mongo_db.conversations, query, "updated_at", -1, limit + 1)
    )
    if documents is None:
        raise AppError("Conversation history is unavailable right now.", status_code=503, code="memory_unavailable")
    has_more = len(documents) > limit
    documents = documents[:limit]
    return ConversationListResponse(
        conversations=[_summary(doc) for doc in documents],
        next_before=documents[-1]["updated_at"] if has_more else None,
    )


async def get_conversation_page(
    runtime: Runtime, client_id: str, conversation_id: str, limit: int, before: float | None
) -> ConversationResponse:
    """Returns a conversation with its latest messages, oldest first.

    Args:
        runtime: Shared runtime.
        client_id: The anonymous browser id.
        conversation_id: The conversation to read.
        limit: Maximum number of messages to return.
        before: Only messages created before this time, to load older history.

    Returns:
        The conversation, a page of messages and the cursor for older messages.

    Raises:
        AppError: With status 404 when the conversation is not found for this
            client, or 503 when the database is unavailable.
    """
    _require_memory(runtime)
    db = runtime.mongo_db
    conversation = await _read(
        runtime, "conversation.get", db.conversations.find_one({"_id": conversation_id, "client_id": client_id})
    )
    if conversation is None:
        raise AppError("Conversation not found", status_code=404, code="conversation_not_found")
    query: dict[str, Any] = {"conversation_id": conversation_id, "client_id": client_id}
    if before is not None:
        query["created_at"] = {"$lt": before}
    documents = await _read(runtime, "messages.list", _fetch(db.messages, query, "created_at", -1, limit + 1))
    if documents is None:
        raise AppError("Conversation history is unavailable right now.", status_code=503, code="memory_unavailable")
    has_more = len(documents) > limit
    documents = documents[:limit]
    documents.reverse()
    return ConversationResponse(
        conversation=_summary(conversation),
        messages=[_message_response(doc) for doc in documents],
        has_more=has_more,
        next_before=documents[0]["created_at"] if has_more and documents else None,
    )


async def delete_conversation(runtime: Runtime, client_id: str, conversation_id: str) -> int:
    """Deletes a conversation with all its messages and jobs.

    Args:
        runtime: Shared runtime.
        client_id: The anonymous browser id.
        conversation_id: The conversation to delete.

    Returns:
        The number of messages deleted.

    Raises:
        AppError: With status 404 when the conversation is not found for this
            client, or 503 when the database is unavailable.
    """
    _require_memory(runtime)
    db = runtime.mongo_db
    scope = {"conversation_id": conversation_id, "client_id": client_id}
    found = await _read(
        runtime, "conversation.get", db.conversations.find_one({"_id": conversation_id, "client_id": client_id})
    )
    if found is None:
        raise AppError("Conversation not found", status_code=404, code="conversation_not_found")
    deleted = await _read(runtime, "messages.delete", db.messages.delete_many(scope))
    await _read(runtime, "jobs.delete", db.jobs.delete_many(scope))
    await _read(runtime, "conversation.delete", db.conversations.delete_one({"_id": conversation_id, "client_id": client_id}))
    logger.info("memory.conversation_deleted conversation_id=%s", conversation_id)
    return deleted.deleted_count if deleted else 0


async def lookup_id(runtime: Runtime, client_id: str, identifier: str) -> LookupResponse:
    """Finds a conversation from any of its ids.

    Tries a conversation id, a job id, a message id and a request id, in that
    order. Only the owner's documents match.

    Args:
        runtime: Shared runtime.
        client_id: The anonymous browser id.
        identifier: Any id to look up.

    Returns:
        What matched and the related ids.

    Raises:
        AppError: With status 404 when nothing matches, or 503 when the database
            is unavailable.
    """
    _require_memory(runtime)
    db = runtime.mongo_db
    owner = {"client_id": client_id}

    conversation = await _read(runtime, "lookup.conversation", db.conversations.find_one({"_id": identifier, **owner}))
    if conversation:
        return LookupResponse(matched="conversation", conversation_id=conversation["_id"])

    job = await _read(runtime, "lookup.job", db.jobs.find_one({"_id": identifier, **owner}))
    if job:
        return LookupResponse(
            matched="job",
            conversation_id=job["conversation_id"],
            job_id=job["_id"],
            message_id=job.get("assistant_message_id") or job.get("user_message_id"),
            request_id=job.get("request_id"),
        )

    message = await _read(runtime, "lookup.message", db.messages.find_one({"_id": identifier, **owner}))
    if message:
        return LookupResponse(
            matched="message",
            conversation_id=message["conversation_id"],
            job_id=message.get("job_id"),
            message_id=message["_id"],
            request_id=message.get("request_id"),
        )

    by_request = await _read(runtime, "lookup.request", db.messages.find_one({"request_id": identifier, **owner}))
    if by_request:
        return LookupResponse(
            matched="request",
            conversation_id=by_request["conversation_id"],
            job_id=by_request.get("job_id"),
            message_id=by_request["_id"],
            request_id=identifier,
        )
    raise AppError("Nothing found for that id", status_code=404, code="id_not_found")


async def load_job_response(runtime: Runtime, job_id: str, client_id: str) -> JobResponse | None:
    """Rebuilds a job's public view from MongoDB, for jobs no longer in memory.

    Args:
        runtime: Shared runtime.
        job_id: The job identifier.
        client_id: The anonymous browser id, which must own the job.

    Returns:
        The job view, or None when it is not found for this client.
    """
    if not memory_available(runtime):
        return None
    db = runtime.mongo_db
    job = await _read(runtime, "job.get", db.jobs.find_one({"_id": job_id, "client_id": client_id}))
    if job is None:
        return None
    assistant = None
    if job.get("assistant_message_id"):
        assistant = await _read(runtime, "job.message", db.messages.find_one({"_id": job["assistant_message_id"]}))

    result = None
    reply = None
    if assistant:
        message = _message_response(assistant)
        result = message.result
        if message.kind == "reply" and message.intent:
            reply = ChatReply(intent=Intent(message.intent), text=message.text)
    error = JobError.model_validate(job["error"]) if job.get("error") else None
    return JobResponse(
        job_id=job["_id"],
        status=JobStatus(job["status"]),
        query=job.get("query", ""),
        conversation_id=job.get("conversation_id"),
        created_at=job["created_at"],
        finished_at=job.get("finished_at"),
        events_count=0,
        result=result,
        reply=reply,
        error=error,
    )