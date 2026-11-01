import asyncio
import os
import time

os.environ.setdefault("GROQ_API_KEY", "test-groq")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily")
os.environ.setdefault("GITHUB_TOKEN", "test-github")
os.environ.setdefault("MONGODB_ENABLED", "false")

import pytest
import pytest_asyncio

from app.controllers import job_controller, memory_controller
from app.controllers.writer_controller import render_brief_markdown, sanitize_citations
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.mongo import connect_mongo
from app.core.runtime import close_runtime, create_runtime
from app.schemas.chat import ChatReply, Intent
from app.schemas.jobs import JobError, JobRecord, JobStatus
from app.schemas.research import ResearchResponse
from tests.test_unit import make_brief, make_sources

TEST_DB = "researchmind_test"
OWNER = "a" * 32
STRANGER = "b" * 32


@pytest_asyncio.fixture
async def mongo_runtime():
    settings = get_settings().model_copy(
        update={"mongodb_enabled": True, "mongodb_db": TEST_DB, "mongodb_timeout_ms": 1500}
    )
    runtime = create_runtime(settings)
    await connect_mongo(runtime)
    if runtime.mongo_db is None:
        await close_runtime(runtime)
        pytest.skip("MongoDB is not running on localhost")
    yield runtime
    await runtime.mongo_client.drop_database(TEST_DB)
    await close_runtime(runtime)


def make_job(conversation_id: str, client_id: str = OWNER, job_id: str | None = None) -> JobRecord:
    return JobRecord(
        id=job_id or memory_controller.new_id(),
        query="evaluate vector databases",
        client_id=client_id,
        conversation_id=conversation_id,
        request_id="req-" + memory_controller.new_id()[:8],
        created_at=time.time(),
        condition=asyncio.Condition(),
    )


def make_result() -> ResearchResponse:
    brief = sanitize_citations(make_brief(), source_count=2)
    sources = make_sources(2)
    return ResearchResponse(
        query="evaluate vector databases",
        sub_questions=["a"],
        sources=[
            {"id": i, "title": s["title"], "url": s["url"], "source": s["source"], "relevance_score": s["relevance_score"]}
            for i, s in enumerate(sources, 1)
        ],
        brief=brief,
        report=render_brief_markdown(brief, sources),
        source_count=2,
        duration_seconds=1.0,
    )


async def start_job(runtime, conversation_id: str, job_id: str | None = None):
    job = make_job(conversation_id, job_id=job_id)
    acquired = await memory_controller.acquire_conversation(runtime, conversation_id, OWNER, job.id)
    assert acquired is True
    job.user_message_id = await memory_controller.record_job_start(
        runtime, job.id, conversation_id, OWNER, job.request_id, job.query
    )
    assert job.user_message_id is not None
    return job


@pytest.mark.asyncio
async def test_indexes_include_ttl_and_unique_job_role(mongo_runtime):
    messages = await mongo_runtime.mongo_db.messages.index_information()
    conversations = await mongo_runtime.mongo_db.conversations.index_information()
    assert messages["ttl_expires"]["expireAfterSeconds"] == 0
    assert messages["job_role_unique"]["unique"] is True
    assert conversations["ttl_expires"]["expireAfterSeconds"] == 0


@pytest.mark.asyncio
async def test_conversation_is_created_once_per_owner_and_reused(mongo_runtime):
    created = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "evaluate vector databases")
    reused = await memory_controller.ensure_conversation(mongo_runtime, OWNER, created["_id"], "another message")
    stolen = await memory_controller.ensure_conversation(mongo_runtime, STRANGER, created["_id"], "my message")
    assert reused["_id"] == created["_id"]
    assert stolen["_id"] != created["_id"]
    assert created["title"] == "evaluate vector databases"


@pytest.mark.asyncio
async def test_only_one_job_can_hold_a_conversation(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    cid = conversation["_id"]
    first = await memory_controller.acquire_conversation(mongo_runtime, cid, OWNER, "job-one")
    second = await memory_controller.acquire_conversation(mongo_runtime, cid, OWNER, "job-two")
    await memory_controller.release_conversation(mongo_runtime, cid, "job-one")
    third = await memory_controller.acquire_conversation(mongo_runtime, cid, OWNER, "job-two")
    assert (first, second, third) == (True, False, True)


@pytest.mark.asyncio
async def test_a_busy_conversation_rejects_a_second_message_with_409(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    await memory_controller.acquire_conversation(mongo_runtime, conversation["_id"], OWNER, "job-one")
    with pytest.raises(AppError) as exc_info:
        await job_controller.submit_chat_job(
            mongo_runtime, "another question here", OWNER, conversation["_id"], "req-1"
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "conversation_busy"


@pytest.mark.asyncio
async def test_finished_research_job_is_saved_and_found_by_every_id(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "evaluate vector databases")
    cid = conversation["_id"]
    job = await start_job(mongo_runtime, cid)
    job.result = make_result()
    await memory_controller.persist_outcome(mongo_runtime, job, JobStatus.COMPLETED)

    by_conversation = await memory_controller.lookup_id(mongo_runtime, OWNER, cid)
    by_job = await memory_controller.lookup_id(mongo_runtime, OWNER, job.id)
    by_request = await memory_controller.lookup_id(mongo_runtime, OWNER, job.request_id)
    by_message = await memory_controller.lookup_id(mongo_runtime, OWNER, by_job.message_id)
    assert by_conversation.matched == "conversation"
    assert by_job.matched == "job" and by_job.conversation_id == cid
    assert by_request.matched == "request" and by_request.job_id == job.id
    assert by_message.matched == "message" and by_message.conversation_id == cid

    page = await memory_controller.get_conversation_page(mongo_runtime, OWNER, cid, 50, None)
    assert [m.role for m in page.messages] == ["user", "assistant"]
    assert page.messages[1].kind == "research"
    assert page.messages[1].result.brief.recommendation.choice == "Qdrant"
    assert page.conversation.message_count == 2
    assert page.conversation.active_job_id is None


@pytest.mark.asyncio
async def test_lookups_never_cross_clients(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    job = await start_job(mongo_runtime, conversation["_id"])
    for identifier in (conversation["_id"], job.id, job.request_id):
        with pytest.raises(AppError) as exc_info:
            await memory_controller.lookup_id(mongo_runtime, STRANGER, identifier)
        assert exc_info.value.status_code == 404
    with pytest.raises(AppError):
        await memory_controller.get_conversation_page(mongo_runtime, STRANGER, conversation["_id"], 10, None)
    assert await memory_controller.load_job_response(mongo_runtime, job.id, STRANGER) is None


@pytest.mark.asyncio
async def test_saving_the_same_job_twice_does_not_duplicate_the_answer(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    job = await start_job(mongo_runtime, conversation["_id"])
    job.reply = ChatReply(intent=Intent.GREETING, text="Hello!")
    await memory_controller.persist_outcome(mongo_runtime, job, JobStatus.COMPLETED)
    await memory_controller.persist_outcome(mongo_runtime, job, JobStatus.COMPLETED)
    count = await mongo_runtime.mongo_db.messages.count_documents({"job_id": job.id, "role": "assistant"})
    assert count == 1


@pytest.mark.asyncio
async def test_failed_and_cancelled_jobs_keep_their_partial_text(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    cid = conversation["_id"]
    failed = await start_job(mongo_runtime, cid)
    failed.stream_text = "Qdrant is"
    failed.error = JobError(code="boom", message="Model failed", status_code=502)
    await memory_controller.persist_outcome(mongo_runtime, failed, JobStatus.FAILED)
    cancelled = await start_job(mongo_runtime, cid)
    cancelled.stream_text = "Partial answer"
    await memory_controller.persist_outcome(mongo_runtime, cancelled, JobStatus.CANCELLED)

    page = await memory_controller.get_conversation_page(mongo_runtime, OWNER, cid, 50, None)
    assistants = [m for m in page.messages if m.role == "assistant"]
    assert [(m.kind, m.status) for m in assistants] == [("error", "failed"), ("cancelled", "cancelled")]
    assert assistants[0].text == "Qdrant is"
    assert assistants[1].text == "Partial answer"


@pytest.mark.asyncio
async def test_context_includes_recent_turns_and_the_latest_brief_summary(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "evaluate vector databases")
    cid = conversation["_id"]
    job = await start_job(mongo_runtime, cid)
    job.result = make_result()
    await memory_controller.persist_outcome(mongo_runtime, job, JobStatus.COMPLETED)

    context = await memory_controller.load_context(mongo_runtime, cid, OWNER)
    assert context.turns[0] == "User: evaluate vector databases"
    assert "recommended: Qdrant" in context.turns[1]
    assert "Recommendation: Qdrant" in context.brief_summary
    assert "[1] Source 1" in context.brief_summary


@pytest.mark.asyncio
async def test_refused_exchanges_are_left_out_of_the_context(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    cid = conversation["_id"]
    job = await start_job(mongo_runtime, cid)
    job.reply = ChatReply(intent=Intent.UNSAFE, text="I can't help with that.")
    await memory_controller.persist_outcome(mongo_runtime, job, JobStatus.COMPLETED)
    context = await memory_controller.load_context(mongo_runtime, cid, OWNER)
    assert context.turns == []
    assert context.brief_summary is None


@pytest.mark.asyncio
async def test_conversations_list_newest_first_with_a_cursor(mongo_runtime):
    ids = []
    for index in range(3):
        conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, f"topic number {index}")
        ids.append(conversation["_id"])
        await asyncio.sleep(0.01)
    first = await memory_controller.list_conversations(mongo_runtime, OWNER, 2, None)
    second = await memory_controller.list_conversations(mongo_runtime, OWNER, 2, first.next_before)
    assert [c.conversation_id for c in first.conversations] == [ids[2], ids[1]]
    assert first.next_before is not None
    assert [c.conversation_id for c in second.conversations] == [ids[0]]
    assert second.next_before is None


@pytest.mark.asyncio
async def test_message_pages_load_older_history_with_a_cursor(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    cid = conversation["_id"]
    for index in range(3):
        job = await start_job(mongo_runtime, cid)
        job.reply = ChatReply(intent=Intent.GREETING, text=f"reply {index}")
        await memory_controller.persist_outcome(mongo_runtime, job, JobStatus.COMPLETED)
    latest = await memory_controller.get_conversation_page(mongo_runtime, OWNER, cid, 4, None)
    older = await memory_controller.get_conversation_page(mongo_runtime, OWNER, cid, 4, latest.next_before)
    assert len(latest.messages) == 4 and latest.has_more is True
    assert len(older.messages) == 2 and older.has_more is False
    assert older.messages[-1].created_at < latest.messages[0].created_at


@pytest.mark.asyncio
async def test_recovery_marks_running_jobs_interrupted_and_clears_the_lock(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "evaluate vector databases")
    cid = conversation["_id"]
    job = await start_job(mongo_runtime, cid)
    recovered = await memory_controller.recover_interrupted_jobs(mongo_runtime)
    view = await memory_controller.load_job_response(mongo_runtime, job.id, OWNER)
    page = await memory_controller.get_conversation_page(mongo_runtime, OWNER, cid, 50, None)
    assert recovered == 1
    assert view.status == JobStatus.INTERRUPTED
    assert view.error.code == "interrupted"
    assert page.messages[-1].status == "interrupted"
    assert page.conversation.active_job_id is None


@pytest.mark.asyncio
async def test_stored_jobs_are_served_after_they_leave_memory(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    job = await start_job(mongo_runtime, conversation["_id"])
    job.reply = ChatReply(intent=Intent.GREETING, text="Hello!")
    await memory_controller.persist_outcome(mongo_runtime, job, JobStatus.COMPLETED)
    view = await job_controller.get_job_view(mongo_runtime, job.id, OWNER)
    frames = [frame async for frame in job_controller.stream_stored_job(view)]
    assert view.status == JobStatus.COMPLETED
    assert view.reply.text == "Hello!"
    assert "event: job_completed" in frames[0]
    assert '"reply"' in frames[0]


@pytest.mark.asyncio
async def test_deleting_a_conversation_removes_messages_and_jobs(mongo_runtime):
    conversation = await memory_controller.ensure_conversation(mongo_runtime, OWNER, None, "hello there")
    cid = conversation["_id"]
    job = await start_job(mongo_runtime, cid)
    job.reply = ChatReply(intent=Intent.GREETING, text="Hello!")
    await memory_controller.persist_outcome(mongo_runtime, job, JobStatus.COMPLETED)
    deleted = await memory_controller.delete_conversation(mongo_runtime, OWNER, cid)
    assert deleted == 2
    assert await mongo_runtime.mongo_db.jobs.count_documents({"conversation_id": cid}) == 0
    with pytest.raises(AppError) as exc_info:
        await memory_controller.get_conversation_page(mongo_runtime, OWNER, cid, 10, None)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_submit_chat_job_saves_the_message_and_returns_the_conversation(mongo_runtime, monkeypatch):
    from unittest.mock import AsyncMock

    from app.schemas.chat import RouteDecision

    monkeypatch.setattr(job_controller, "route_message", AsyncMock(return_value=RouteDecision(intent=Intent.GREETING, reply="Hello!")))
    job = await job_controller.submit_chat_job(mongo_runtime, "hello there friend", OWNER, None, "req-abc")
    await job_controller._wait_until_finished(job)
    assert job.conversation_id is not None
    assert job.user_message_id is not None
    page = await memory_controller.get_conversation_page(mongo_runtime, OWNER, job.conversation_id, 50, None)
    assert [m.role for m in page.messages] == ["user", "assistant"]
    assert page.messages[0].text == "hello there friend"
    assert page.messages[1].text == "Hello!"
    assert page.conversation.active_job_id is None