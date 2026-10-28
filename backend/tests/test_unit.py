import asyncio
import os
import time

os.environ.setdefault("GROQ_API_KEY", "test-groq")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily")
os.environ.setdefault("GITHUB_TOKEN", "test-github")

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.controllers import (
    job_controller,
    planner_controller,
    search_controller,
    writer_controller,
)
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.runtime import close_runtime, create_runtime
from app.main import app
from app.schemas.jobs import JobStatus
from app.schemas.research import (
    ComparisonEntry,
    Confidence,
    Maturity,
    RDBrief,
    Recommendation,
    ResearchResponse,
    Risk,
    Severity,
    TechOption,
)


def make_brief(confidence: Confidence = Confidence.HIGH, source_ids: list[int] | None = None) -> RDBrief:
    ids = source_ids if source_ids is not None else [1, 2]
    return RDBrief(
        title="Vector DB [2026] evaluation",
        problem_statement="Pick a vector database for 1M documents.",
        executive_summary="Qdrant fits best.",
        landscape="Several mature options exist.",
        options=[
            TechOption(
                name="Qdrant",
                summary="Rust-based vector database.",
                strengths=["fast filtering"],
                weaknesses=["smaller ecosystem"],
                maturity=Maturity.GROWING,
                source_ids=ids + [9],
            ),
            TechOption(name="FAISS", summary="Library, not a server.", source_ids=[2]),
        ],
        comparison=[
            ComparisonEntry(criterion="Scalability", option="Qdrant", assessment="Horizontal", source_ids=ids + [7]),
            ComparisonEntry(criterion="Scalability", option="FAISS", assessment="Single node", source_ids=[2]),
        ],
        risks=[Risk(description="Vendor lock-in", severity=Severity.LOW, mitigation="Use open source")],
        recommendation=Recommendation(
            choice="Qdrant",
            rationale="Best balance.",
            confidence=confidence,
            next_steps=["Run a benchmark"],
        ),
        poc_plan=["Index 100k docs", "Measure p95 latency"],
        limitations=["No cost data found"],
    )


def make_sources(count: int = 2, same_domain: bool = False) -> list[dict]:
    return [
        {
            "sub_question": "q",
            "source": "tavily" if i % 2 else "github",
            "title": f"Source {i}",
            "url": "https://example.com/a" if same_domain else f"https://site{i}.com/{i}",
            "content": "snippet",
            "relevance_score": 1 - i / 10,
        }
        for i in range(1, count + 1)
    ]


def make_response() -> ResearchResponse:
    brief = writer_controller.sanitize_citations(make_brief(), source_count=2)
    sources = make_sources(2)
    return ResearchResponse(
        query="evaluate vector databases",
        sub_questions=["a"],
        sources=[
            {"id": i, "title": s["title"], "url": s["url"], "source": s["source"], "relevance_score": s["relevance_score"]}
            for i, s in enumerate(sources, 1)
        ],
        brief=brief,
        report=writer_controller.render_brief_markdown(brief, sources),
        source_count=2,
        duration_seconds=1.2,
    )


def fake_llm(side_effect) -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=side_effect)
    return llm


async def with_runtime(settings=None):
    return create_runtime(settings or get_settings())


def runtime_config(runtime) -> dict:
    return {"configurable": {"runtime": runtime}}


def test_tavily_results_use_real_scores_and_drop_low_relevance():
    items = [
        {"title": "Good", "url": "https://a.com", "content": "text", "score": 0.91},
        {"title": "Weak", "url": "https://b.com", "content": "text", "score": 0.05},
        {"title": "Bad url", "url": "ftp://c.com", "content": "text", "score": 0.9},
        {"title": "No score", "url": "https://d.com", "content": "text"},
    ]
    results = search_controller._tavily_to_results(items, "q", min_score=0.2)
    assert [r["title"] for r in results] == ["Good"]
    assert results[0]["relevance_score"] == 0.91
    assert results[0]["source"] == "tavily"


def test_github_results_include_health_signals_and_filter_low_stars():
    items = [
        {
            "full_name": "a/good",
            "html_url": "https://github.com/a/good",
            "description": "ok",
            "stargazers_count": 5000,
            "license": {"spdx_id": "Apache-2.0"},
            "pushed_at": "2026-10-01T10:00:00Z",
        },
        {"full_name": "a/low", "html_url": "https://github.com/a/low", "description": "ok", "stargazers_count": 3},
        {"full_name": "a/nodesc", "html_url": "https://github.com/a/nodesc", "description": None, "stargazers_count": 900},
    ]
    results = search_controller._github_to_results(items, "topic")
    assert [r["title"] for r in results] == ["a/good"]
    assert "license: Apache-2.0" in results[0]["content"]
    assert "last push: 2026-10-01" in results[0]["content"]
    assert results[0]["relevance_score"] == 1.0


def test_build_github_query_uses_topic_keywords_not_question_words():
    query = search_controller._build_github_query(
        "Which vector database should we use for 1M documents with hybrid search?"
    )
    assert query == "vector database documents hybrid stars:>50"


@pytest.mark.asyncio
async def test_score_sources_dedupes_by_url_and_ranks():
    state = {
        "search_results": [
            {"url": "https://a.com", "relevance_score": 0.5},
            {"url": "https://a.com", "relevance_score": 0.9},
            {"url": "https://b.com", "relevance_score": 0.7},
            {"url": "", "relevance_score": 1.0},
        ]
    }
    result = await search_controller.score_sources(state, {})
    assert [s["url"] for s in result["sources"]] == ["https://a.com", "https://b.com"]
    assert result["sources"][0]["relevance_score"] == 0.9


@pytest.mark.asyncio
async def test_score_sources_raises_when_nothing_was_found():
    with pytest.raises(AppError) as exc_info:
        await search_controller.score_sources({"search_results": []}, {})
    assert exc_info.value.code == "no_sources_found"
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_search_web_runs_sub_questions_in_parallel(monkeypatch):
    async def slow_search(client, query, max_results):
        await asyncio.sleep(0.2)
        return [{"title": f"T {query}", "url": f"https://x.com/{query}", "content": "c", "score": 0.8}]

    monkeypatch.setattr(search_controller, "tavily_search", slow_search)
    runtime = await with_runtime()
    try:
        started = time.perf_counter()
        result = await search_controller.search_web(
            {"sub_questions": ["q1", "q2", "q3", "q4"]}, runtime_config(runtime)
        )
        elapsed = time.perf_counter() - started
    finally:
        await close_runtime(runtime)
    assert len(result["search_results"]) == 4
    assert elapsed < 0.6


@pytest.mark.asyncio
async def test_search_web_survives_a_failed_sub_question(monkeypatch):
    async def flaky_search(client, query, max_results):
        if query == "bad":
            raise httpx.ConnectError("boom")
        return [{"title": "T", "url": f"https://x.com/{query}", "content": "c", "score": 0.8}]

    monkeypatch.setattr(search_controller, "tavily_search", flaky_search)
    runtime = await with_runtime()
    try:
        result = await search_controller.search_web(
            {"sub_questions": ["ok", "bad"]}, runtime_config(runtime)
        )
    finally:
        await close_runtime(runtime)
    assert [r["url"] for r in result["search_results"]] == ["https://x.com/ok"]


@pytest.mark.asyncio
async def test_search_github_returns_empty_on_http_error(monkeypatch):
    monkeypatch.setattr(search_controller, "github_search_repositories", AsyncMock(side_effect=httpx.ReadTimeout("slow")))
    runtime = await with_runtime()
    try:
        result = await search_controller.search_github({"query": "vector database"}, runtime_config(runtime))
    finally:
        await close_runtime(runtime)
    assert result == {"search_results": []}


def test_sanitize_citations_drops_invented_source_numbers():
    brief = writer_controller.sanitize_citations(make_brief(), source_count=2)
    assert brief.options[0].source_ids == [1, 2]
    assert brief.comparison[0].source_ids == [1, 2]


def test_cap_confidence_limits_single_domain_to_medium():
    brief = writer_controller.sanitize_citations(make_brief(Confidence.HIGH), source_count=2)
    brief = writer_controller.cap_confidence(brief, make_sources(2, same_domain=True))
    assert brief.recommendation.confidence == Confidence.MEDIUM
    assert any("Confidence capped" in item for item in brief.limitations)


def test_cap_confidence_keeps_high_with_independent_domains():
    brief = writer_controller.sanitize_citations(make_brief(Confidence.HIGH), source_count=2)
    brief = writer_controller.cap_confidence(brief, make_sources(2))
    assert brief.recommendation.confidence == Confidence.HIGH


def test_cap_confidence_limits_to_low_without_supporting_sources():
    brief = make_brief(Confidence.HIGH, source_ids=[])
    brief.options[0].source_ids = []
    brief.options[1].source_ids = []
    brief.comparison = []
    brief = writer_controller.cap_confidence(brief, make_sources(2))
    assert brief.recommendation.confidence == Confidence.LOW


def test_render_brief_markdown_uses_supported_syntax_and_clean_titles():
    brief = writer_controller.sanitize_citations(make_brief(), source_count=2)
    sources = make_sources(2)
    sources[0]["title"] = "Best Models Compared [2026]"
    markdown = writer_controller.render_brief_markdown(brief, sources)
    assert "## Recommendation" in markdown
    assert "### Scalability" in markdown
    assert "* **Qdrant:** Horizontal [1][2]" in markdown
    assert "* [1] [Best Models Compared (2026)](https://site1.com/1)" in markdown
    assert "|" not in markdown
    assert not markdown.startswith("# ")


def test_brief_schema_tolerates_string_lists_and_uppercase_enums():
    data = make_brief().model_dump(mode="json")
    data["limitations"] = "No cost data found"
    data["poc_plan"] = "Step one\nStep two"
    data["options"][0]["maturity"] = "Mature"
    data["options"][0]["source_ids"] = ["source 1", 2]
    data["recommendation"]["confidence"] = "HIGH"
    brief = RDBrief.model_validate(data)
    assert brief.limitations == ["No cost data found"]
    assert brief.poc_plan == ["Step one", "Step two"]
    assert brief.options[0].maturity == Maturity.MATURE
    assert brief.options[0].source_ids == [1, 2]
    assert brief.recommendation.confidence == Confidence.HIGH


@pytest.mark.asyncio
async def test_decompose_query_falls_back_to_raw_query_on_llm_failure(monkeypatch):
    monkeypatch.setattr(planner_controller, "get_llm", lambda **kwargs: fake_llm(RuntimeError("boom")))
    runtime = await with_runtime()
    try:
        result = await planner_controller.decompose_query(
            {"query": "evaluate vector databases"}, runtime_config(runtime)
        )
    finally:
        await close_runtime(runtime)
    assert result == {"sub_questions": ["evaluate vector databases"]}


@pytest.mark.asyncio
async def test_synthesize_brief_retries_then_succeeds(monkeypatch):
    llm = fake_llm([RuntimeError("bad output"), make_brief(Confidence.HIGH)])
    monkeypatch.setattr(writer_controller, "get_llm", lambda **kwargs: llm)
    runtime = await with_runtime()
    try:
        state = {"query": "q", "sub_questions": ["a"], "sources": make_sources(2)}
        result = await writer_controller.synthesize_brief(state, runtime_config(runtime))
    finally:
        await close_runtime(runtime)
    assert result["brief"].options[0].source_ids == [1, 2]
    assert "## Executive Summary" in result["report"]


@pytest.mark.asyncio
async def test_synthesize_brief_raises_app_error_after_all_attempts(monkeypatch):
    llm = fake_llm(RuntimeError("always fails"))
    monkeypatch.setattr(writer_controller, "get_llm", lambda **kwargs: llm)
    runtime = await with_runtime()
    try:
        state = {"query": "q", "sub_questions": [], "sources": make_sources(2)}
        with pytest.raises(AppError) as exc_info:
            await writer_controller.synthesize_brief(state, runtime_config(runtime))
    finally:
        await close_runtime(runtime)
    assert exc_info.value.code == "brief_generation_failed"
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_job_completes_and_streams_events_in_order(monkeypatch):
    async def fake_run(runtime, query, emit):
        await emit("node_started", {"node": "decompose_query"})
        await emit("node_completed", {"node": "decompose_query"})
        return make_response()

    monkeypatch.setattr(job_controller, "run_research", fake_run)
    runtime = await with_runtime()
    try:
        job = job_controller.submit_job(runtime, "evaluate vector databases")
        frames = [frame async for frame in job_controller.stream_job_events(runtime, job, 0)]
    finally:
        await close_runtime(runtime)
    names = [line.split(": ", 1)[1] for frame in frames for line in frame.splitlines() if line.startswith("event: ")]
    assert names == ["job_queued", "job_started", "node_started", "node_completed", "job_completed"]
    assert job.status == JobStatus.COMPLETED
    assert job.result is not None


@pytest.mark.asyncio
async def test_stream_resumes_from_last_event_id(monkeypatch):
    monkeypatch.setattr(job_controller, "run_research", AsyncMock(return_value=make_response()))
    runtime = await with_runtime()
    try:
        job = job_controller.submit_job(runtime, "evaluate vector databases")
        await job_controller._wait_until_finished(job)
        frames = [frame async for frame in job_controller.stream_job_events(runtime, job, 2)]
    finally:
        await close_runtime(runtime)
    assert len(frames) == len(job.events) - 2
    assert "event: job_completed" in frames[-1]


@pytest.mark.asyncio
async def test_job_failure_records_app_error(monkeypatch):
    monkeypatch.setattr(
        job_controller,
        "run_research",
        AsyncMock(side_effect=AppError("No sources", status_code=502, code="no_sources_found")),
    )
    runtime = await with_runtime()
    try:
        job = job_controller.submit_job(runtime, "evaluate vector databases")
        await job_controller._wait_until_finished(job)
    finally:
        await close_runtime(runtime)
    assert job.status == JobStatus.FAILED
    assert job.error.code == "no_sources_found"


@pytest.mark.asyncio
async def test_job_can_be_cancelled(monkeypatch):
    async def slow_run(runtime, query, emit):
        await asyncio.sleep(30)

    monkeypatch.setattr(job_controller, "run_research", slow_run)
    runtime = await with_runtime()
    try:
        job = job_controller.submit_job(runtime, "evaluate vector databases")
        await asyncio.sleep(0.05)
        cancelled = await job_controller.cancel_job(runtime, job.id)
    finally:
        await close_runtime(runtime)
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.events[-1].event == "job_cancelled"


@pytest.mark.asyncio
async def test_submit_rejects_when_too_many_jobs_are_pending(monkeypatch):
    async def slow_run(runtime, query, emit):
        await asyncio.sleep(30)

    monkeypatch.setattr(job_controller, "run_research", slow_run)
    runtime = await with_runtime(get_settings().model_copy(update={"max_pending_jobs": 1}))
    try:
        job_controller.submit_job(runtime, "evaluate vector databases")
        with pytest.raises(AppError) as exc_info:
            job_controller.submit_job(runtime, "evaluate another topic")
    finally:
        await close_runtime(runtime)
    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "server_busy"


@pytest.mark.asyncio
async def test_pipeline_concurrency_is_bounded(monkeypatch):
    running = 0
    peak = 0

    async def tracked_run(runtime, query, emit):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.1)
        running -= 1
        return make_response()

    monkeypatch.setattr(job_controller, "run_research", tracked_run)
    runtime = await with_runtime(get_settings().model_copy(update={"max_concurrent_pipelines": 2}))
    try:
        jobs = [job_controller.submit_job(runtime, f"evaluate topic number {i}") for i in range(5)]
        for job in jobs:
            await job_controller._wait_until_finished(job)
    finally:
        await close_runtime(runtime)
    assert peak == 2
    assert all(job.status == JobStatus.COMPLETED for job in jobs)


def test_purge_removes_expired_finished_jobs():
    async def scenario():
        runtime = await with_runtime(get_settings().model_copy(update={"job_ttl_seconds": 1}))
        try:
            job = job_controller.submit_job(runtime, "evaluate vector databases")
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
            job.finished_at = time.time() - 10
            removed = job_controller.purge_expired_jobs(runtime)
            return removed, len(runtime.jobs)
        finally:
            await close_runtime(runtime)

    removed, remaining = asyncio.run(scenario())
    assert removed == 1
    assert remaining == 0


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def wait_for_status(client: TestClient, url: str, expected: str, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    body: dict = {}
    while time.time() < deadline:
        body = client.get(url).json()
        if body["status"] == expected:
            return body
        time.sleep(0.05)
    return body


def test_api_returns_error_envelope_for_app_error(client, monkeypatch):
    monkeypatch.setattr(
        "app.routers.research.run_research_blocking",
        AsyncMock(side_effect=AppError("No sources", status_code=502, code="no_sources_found")),
    )
    response = client.post("/api/research", json={"query": "evaluate vector databases"})
    body = response.json()
    assert response.status_code == 502
    assert body["detail"] == "No sources"
    assert body["error"]["code"] == "no_sources_found"
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]


def test_api_hides_internals_on_unexpected_error(monkeypatch):
    monkeypatch.setattr(
        "app.routers.research.run_research_blocking",
        AsyncMock(side_effect=RuntimeError("secret internal detail")),
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/api/research", json={"query": "evaluate vector databases"})
    assert response.status_code == 500
    assert "secret" not in response.text


def test_api_validates_short_query(client):
    response = client.post("/api/research", json={"query": "short"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_api_blocking_endpoint_returns_brief_and_markdown(client, monkeypatch):
    monkeypatch.setattr("app.routers.research.run_research_blocking", AsyncMock(return_value=make_response()))
    response = client.post("/api/research", json={"query": "evaluate vector databases"})
    body = response.json()
    assert response.status_code == 200
    assert body["brief"]["recommendation"]["choice"] == "Qdrant"
    assert body["report"].startswith("## Vector DB [2026] evaluation")
    assert body["source_count"] == 2


def test_api_job_flow_create_poll_and_stream_replay(client, monkeypatch):
    async def fake_run(runtime, query, emit):
        await emit("node_started", {"node": "decompose_query"})
        return make_response()

    monkeypatch.setattr(job_controller, "run_research", fake_run)
    created = client.post("/api/research/jobs", json={"query": "evaluate vector databases"})
    assert created.status_code == 202
    body = created.json()
    assert body["status_url"].endswith(body["job_id"])
    assert body["stream_url"].endswith("/stream")

    job = wait_for_status(client, body["status_url"], "completed")
    assert job["status"] == "completed"
    assert job["result"]["brief"]["recommendation"]["choice"] == "Qdrant"

    stream = client.get(body["stream_url"])
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert "event: node_started" in stream.text
    assert "event: job_completed" in stream.text


def test_api_unknown_job_returns_404(client):
    response = client.get("/api/research/jobs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


def test_api_cancel_job(client, monkeypatch):
    async def slow_run(runtime, query, emit):
        await asyncio.sleep(30)

    monkeypatch.setattr(job_controller, "run_research", slow_run)
    created = client.post("/api/research/jobs", json={"query": "evaluate vector databases"}).json()
    response = client.delete(created["status_url"])
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_api_readiness_reports_job_counts(client):
    response = client.get("/api/health/ready")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert set(body) == {"status", "queued_jobs", "running_jobs", "stored_jobs"}