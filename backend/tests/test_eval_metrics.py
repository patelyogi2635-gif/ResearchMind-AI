import json
from pathlib import Path

import pytest

from evals.metrics import (
    MAX_CLAIMS,
    citation_stats,
    extract_claims,
    format_judge_prompt,
    groundedness_score,
    mean,
    normalize_verdicts,
    percentile,
    regressions,
    render_markdown,
    source_domains,
    structure_issues,
    summarize,
    topic_covered,
    unsupported_examples,
)

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden_topics.json"


def sample_brief() -> dict:
    return {
        "executive_summary": "Qdrant fits best.",
        "poc_plan": ["Index 100k docs"],
        "options": [
            {"name": "Qdrant", "source_ids": [1], "strengths": ["fast filtering"], "weaknesses": ["young ecosystem"]},
            {"name": "FAISS", "source_ids": [], "strengths": [], "weaknesses": ["no server"]},
        ],
        "comparison": [
            {"criterion": "Scale", "option": "Qdrant", "assessment": "horizontal", "source_ids": [1, 9]},
            {"criterion": "Scale", "option": "FAISS", "assessment": "single node", "source_ids": []},
        ],
        "recommendation": {"choice": "Qdrant", "rationale": "Best balance.", "confidence": "high"},
    }


def ok_result(latency: float, grounded: float | None, covered: bool | None = True) -> dict:
    return {
        "ok": True,
        "error_code": None,
        "latency_seconds": latency,
        "node_seconds": {"search_web": latency / 2, "synthesize_brief": latency / 4},
        "tokens": {"input": 1000, "output": 500, "total": 1500},
        "structure_issues": [],
        "citation": {"coverage": 0.8},
        "topic_covered": covered,
        "confidence": "medium",
        "groundedness": grounded,
        "sources_count": 6,
        "domains": ["a.com", "b.com"],
    }


def test_percentile_interpolates_and_handles_small_inputs():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert percentile([7], 95) == 7
    assert percentile([], 50) is None
    assert mean([]) is None


def test_structure_issues_lists_what_is_missing():
    assert structure_issues(sample_brief()) == []
    broken = sample_brief() | {"poc_plan": [], "options": sample_brief()["options"][:1]}
    issues = structure_issues(broken)
    assert "fewer than 2 options" in issues
    assert "no proof-of-concept plan" in issues


def test_citation_stats_counts_only_valid_source_numbers():
    stats = citation_stats(sample_brief(), source_count=2)
    assert stats["items"] == 4
    assert stats["cited_items"] == 2
    assert stats["coverage"] == 0.5
    assert stats["distinct_sources_cited"] == 1
    assert citation_stats({"options": [], "comparison": []}, 3)["coverage"] == 0.0


def test_topic_covered_matches_options_and_the_recommendation():
    assert topic_covered(sample_brief(), ["qdrant"]) is True
    assert topic_covered(sample_brief(), ["milvus", "weaviate"]) is False
    assert topic_covered(sample_brief(), []) is None


def test_source_domains_are_distinct_and_without_www():
    sources = [{"url": "https://www.a.com/x"}, {"url": "https://a.com/y"}, {"url": "https://b.org"}, {"url": ""}]
    assert source_domains(sources) == ["a.com", "b.org"]


def test_extract_claims_orders_claims_and_keeps_the_rationale_last():
    claims = extract_claims(sample_brief())
    assert [claim["id"] for claim in claims] == list(range(1, len(claims) + 1))
    assert claims[0]["text"].startswith("Qdrant on Scale")
    assert claims[-1]["text"].startswith("Recommendation (Qdrant)")
    assert len(claims) == 6


def test_extract_claims_is_capped_for_judging_cost():
    brief = sample_brief()
    brief["comparison"] = [
        {"criterion": f"c{i}", "option": "Qdrant", "assessment": "x", "source_ids": []} for i in range(60)
    ]
    claims = extract_claims(brief)
    assert len(claims) == MAX_CLAIMS
    assert claims[-1]["text"].startswith("Recommendation")


def test_judge_prompt_lists_claims_with_their_citations():
    prompt = format_judge_prompt([{"id": 1, "text": "Qdrant is fast", "cited": [1, 2]}], "[1] Source")
    assert "1. Qdrant is fast (cites [1][2])" in prompt
    assert "exactly 1 entries" in prompt


def test_verdicts_are_normalized_and_padded():
    verdicts = normalize_verdicts(["Supported", "partially supported", "no", "banana"], 6)
    assert verdicts == ["supported", "partial", "unsupported", "unsupported", "unjudged", "unjudged"]
    assert normalize_verdicts(["supported", "supported", "supported"], 2) == ["supported", "supported"]


def test_groundedness_ignores_unjudged_claims():
    assert groundedness_score(["supported", "partial", "unsupported", "unjudged"]) == pytest.approx(0.5)
    assert groundedness_score(["unjudged", "unjudged"]) is None
    claims = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert unsupported_examples(claims, ["supported", "unsupported", "unsupported"], limit=1) == ["b"]


def test_summarize_aggregates_successes_failures_and_latency():
    results = [
        ok_result(10, 0.8),
        ok_result(30, 0.6, covered=False),
        {"ok": False, "error_code": "no_sources_found", "latency_seconds": 5},
    ]
    summary = summarize(results)
    assert summary["topics"] == 3 and summary["succeeded"] == 2
    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["groundedness"] == pytest.approx(0.7)
    assert summary["topic_coverage_rate"] == pytest.approx(0.5)
    assert summary["latency_p50_seconds"] == 20
    assert summary["failures"] == {"no_sources_found": 1}
    assert summary["confidence"] == {"medium": 2}
    assert summary["node_seconds_mean"]["search_web"] == pytest.approx(10)
    assert summary["tokens_mean_total"] == 1500


def test_summarize_handles_a_run_where_everything_failed():
    summary = summarize([{"ok": False, "error_code": "boom", "latency_seconds": 1}])
    assert summary["success_rate"] == 0
    assert summary["groundedness"] is None
    assert summary["latency_p95_seconds"] is None


def test_regressions_flag_drops_and_slowdowns_but_not_improvements():
    baseline = {"success_rate": 1.0, "groundedness": 0.8, "citation_coverage": 0.7, "latency_p95_seconds": 20.0}
    worse = {"success_rate": 1.0, "groundedness": 0.6, "citation_coverage": 0.9, "latency_p95_seconds": 40.0}
    found = regressions(worse, baseline)
    assert any(message.startswith("groundedness dropped") for message in found)
    assert any(message.startswith("latency_p95_seconds rose") for message in found)
    assert not any("citation_coverage" in message for message in found)
    assert regressions(baseline, baseline) == []
    assert regressions({"groundedness": None}, baseline) == []


def test_markdown_report_contains_the_headline_metrics():
    summary = summarize([ok_result(12, 0.75)])
    report = render_markdown(summary, {"timestamp": "now", "writer_model": "w", "chat_model": "c", "judge_model": None})
    assert "| Groundedness (LLM judge) | 0.75 |" in report
    assert "| Latency p50 / p95 (s) | 12.0 / 12.0 |" in report
    assert "judge `off`" in report
    assert "| search_web |" in report


def test_golden_topics_are_well_formed():
    topics = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert len(topics) >= 20
    assert len({topic["id"] for topic in topics}) == len(topics)
    for topic in topics:
        assert 10 <= len(topic["query"]) <= 500
        assert topic["category"]
        assert topic["expect_any"] and all(name == name.lower() for name in topic["expect_any"])