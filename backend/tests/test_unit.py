import json
import os

os.environ.setdefault("GROQ_API_KEY", "test-groq")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily")
os.environ.setdefault("GITHUB_TOKEN", "test-github")

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.controllers import planner_controller, search_controller, writer_controller
from app.core.errors import AppError
from app.main import app
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


def make_brief() -> RDBrief:
    return RDBrief(
        title="Vector DB evaluation",
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
                source_ids=[1, 9],
            ),
            TechOption(name="FAISS", summary="Library, not a server.", source_ids=[2]),
        ],
        comparison=[
            ComparisonEntry(criterion="Scalability", option="Qdrant", assessment="Horizontal", source_ids=[1, 7]),
            ComparisonEntry(criterion="Scalability", option="FAISS", assessment="Single node", source_ids=[2]),
        ],
        risks=[Risk(description="Vendor lock-in", severity=Severity.LOW, mitigation="Use open source")],
        recommendation=Recommendation(
            choice="Qdrant",
            rationale="Best balance.",
            confidence=Confidence.MEDIUM,
            next_steps=["Run a benchmark"],
        ),
        poc_plan=["Index 100k docs", "Measure p95 latency"],
        limitations=["No cost data found"],
    )


def make_sources(count: int = 2) -> list[dict]:
    return [
        {
            "sub_question": "q",
            "source": "tavily" if i % 2 else "github",
            "title": f"Source {i}",
            "url": f"https://example.com/{i}",
            "content": "snippet",
            "relevance_score": 1 - i / 10,
        }
        for i in range(1, count + 1)
    ]


def fake_llm(side_effect) -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=side_effect)
    return llm


def test_parse_text_blocks_reads_multiline_content_and_skips_bad_urls():
    text = (
        "Detailed Results:\n\n"
        "Title: A\nURL: https://a.com\nContent: line one\nline two\n\n"
        "Title: B\nURL: not-a-url\nContent: x\n\n"
        "Title: C\nURL: https://c.com\nContent: z"
    )
    results = search_controller._parse_text_blocks(text, "tavily", "q", 0.7)
    assert [r["title"] for r in results] == ["A", "C"]
    assert results[0]["content"] == "line one line two"


def test_parse_github_results_filters_low_stars_and_missing_description():
    payload = {
        "items": [
            {"full_name": "a/good", "html_url": "https://github.com/a/good", "description": "ok", "stargazers_count": 5000},
            {"full_name": "a/low", "html_url": "https://github.com/a/low", "description": "ok", "stargazers_count": 3},
            {"full_name": "a/nodesc", "html_url": "https://github.com/a/nodesc", "description": None, "stargazers_count": 900},
        ]
    }
    results = search_controller._parse_github_results(json.dumps(payload), "q")
    assert [r["title"] for r in results] == ["a/good"]
    assert results[0]["relevance_score"] == 1.0


def test_unwrap_text_handles_adapter_content_blocks():
    raw = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
    assert search_controller._unwrap_text(raw) == "hello\n\nworld"


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
    result = await search_controller.score_sources(state)
    assert [s["url"] for s in result["sources"]] == ["https://a.com", "https://b.com"]
    assert result["sources"][0]["relevance_score"] == 0.9


def test_sanitize_citations_drops_invented_source_numbers():
    brief = writer_controller.sanitize_citations(make_brief(), source_count=2)
    assert brief.options[0].source_ids == [1]
    assert brief.comparison[0].source_ids == [1]


def test_render_brief_markdown_uses_only_frontend_supported_syntax():
    brief = writer_controller.sanitize_citations(make_brief(), source_count=2)
    markdown = writer_controller.render_brief_markdown(brief, make_sources(2))
    assert "## Recommendation" in markdown
    assert "### Scalability" in markdown
    assert "* **Qdrant:** Horizontal [1]" in markdown
    assert "* [1] [Source 1](https://example.com/1)" in markdown
    assert "|" not in markdown
    assert not markdown.startswith("# ")


@pytest.mark.asyncio
async def test_decompose_query_falls_back_to_raw_query_on_llm_failure(monkeypatch):
    monkeypatch.setattr(planner_controller, "get_llm", lambda temperature=0.3: fake_llm(RuntimeError("boom")))
    result = await planner_controller.decompose_query({"query": "evaluate vector databases"})
    assert result == {"sub_questions": ["evaluate vector databases"]}


@pytest.mark.asyncio
async def test_synthesize_brief_retries_then_succeeds(monkeypatch):
    llm = fake_llm([RuntimeError("bad tool call"), make_brief()])
    monkeypatch.setattr(writer_controller, "get_llm", lambda temperature=0.3: llm)
    state = {"query": "q", "sub_questions": ["a"], "sources": make_sources(2)}
    result = await writer_controller.synthesize_brief(state)
    assert result["brief"].options[0].source_ids == [1]
    assert "## Executive Summary" in result["report"]


@pytest.mark.asyncio
async def test_synthesize_brief_raises_app_error_after_all_attempts(monkeypatch):
    llm = fake_llm(RuntimeError("always fails"))
    monkeypatch.setattr(writer_controller, "get_llm", lambda temperature=0.3: llm)
    state = {"query": "q", "sub_questions": [], "sources": make_sources(2)}
    with pytest.raises(AppError) as exc_info:
        await writer_controller.synthesize_brief(state)
    assert exc_info.value.code == "brief_generation_failed"
    assert exc_info.value.status_code == 502


def test_api_returns_error_envelope_for_app_error(monkeypatch):
    monkeypatch.setattr(
        "app.routers.research.run_research",
        AsyncMock(side_effect=AppError("No sources", status_code=502, code="no_sources_found")),
    )
    response = TestClient(app).post("/api/research", json={"query": "evaluate vector databases"})
    body = response.json()
    assert response.status_code == 502
    assert body["detail"] == "No sources"
    assert body["error"]["code"] == "no_sources_found"
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]


def test_api_hides_internals_on_unexpected_error(monkeypatch):
    monkeypatch.setattr(
        "app.routers.research.run_research",
        AsyncMock(side_effect=RuntimeError("secret internal detail")),
    )
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/research", json={"query": "evaluate vector databases"}
    )
    assert response.status_code == 500
    assert "secret" not in response.text


def test_api_validates_short_query():
    response = TestClient(app).post("/api/research", json={"query": "short"})
    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "validation_error"


def test_api_success_returns_brief_and_markdown_report(monkeypatch):
    brief = writer_controller.sanitize_citations(make_brief(), source_count=2)
    sources = make_sources(2)
    payload = ResearchResponse(
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
    monkeypatch.setattr("app.routers.research.run_research", AsyncMock(return_value=payload))
    response = TestClient(app).post("/api/research", json={"query": "evaluate vector databases"})
    body = response.json()
    assert response.status_code == 200
    assert body["brief"]["recommendation"]["choice"] == "Qdrant"
    assert body["report"].startswith("## Vector DB evaluation")
    assert body["source_count"] == 2
