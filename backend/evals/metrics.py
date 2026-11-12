import json
import math
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

MAX_CLAIMS = 24
JUDGED_VERDICTS = ("supported", "partial", "unsupported")
VERDICT_SCORES = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
VERDICT_ALIASES = {
    "supported": "supported",
    "support": "supported",
    "true": "supported",
    "yes": "supported",
    "partial": "partial",
    "partially": "partial",
    "partially supported": "partial",
    "unsupported": "unsupported",
    "not supported": "unsupported",
    "false": "unsupported",
    "no": "unsupported",
}
REGRESSION_TOLERANCE = 0.05
LATENCY_REGRESSION_RATIO = 1.3
HIGHER_IS_BETTER = (
    "success_rate",
    "structure_ok_rate",
    "citation_coverage",
    "topic_coverage_rate",
    "groundedness",
)


class JudgeVerdicts(BaseModel):
    verdicts: list[str] = Field(
        description="One of supported, partial or unsupported for each claim, in order"
    )


def mean(values: list[float]) -> float | None:
    """Returns the arithmetic mean, or None for an empty list."""
    return sum(values) / len(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    """Returns a percentile using linear interpolation, or None for an empty list.

    Args:
        values: The samples.
        pct: Percentile between 0 and 100.

    Returns:
        The interpolated percentile.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def structure_issues(brief: dict[str, Any]) -> list[str]:
    """Lists the ways a brief falls short of the expected R&D structure."""
    issues: list[str] = []
    if len(brief.get("options") or []) < 2:
        issues.append("fewer than 2 options")
    if len(brief.get("comparison") or []) < 2:
        issues.append("fewer than 2 comparison entries")
    if not (brief.get("recommendation") or {}).get("choice"):
        issues.append("no recommendation")
    if not brief.get("executive_summary"):
        issues.append("no executive summary")
    if not brief.get("poc_plan"):
        issues.append("no proof-of-concept plan")
    return issues


def citation_stats(brief: dict[str, Any], source_count: int) -> dict[str, Any]:
    """Measures how much of the brief is backed by a valid source number.

    Args:
        brief: The brief as a dict.
        source_count: How many numbered sources the brief could cite.

    Returns:
        The number of options and comparison entries, how many carry at least one
        valid citation, the coverage ratio and the distinct sources cited.
    """
    groups = [option.get("source_ids") or [] for option in brief.get("options") or []]
    groups += [entry.get("source_ids") or [] for entry in brief.get("comparison") or []]
    valid_groups = [[i for i in ids if 1 <= i <= source_count] for ids in groups]
    cited = sum(1 for ids in valid_groups if ids)
    return {
        "items": len(groups),
        "cited_items": cited,
        "coverage": cited / len(groups) if groups else 0.0,
        "distinct_sources_cited": len({i for ids in valid_groups for i in ids}),
    }


def topic_covered(brief: dict[str, Any], expect_any: list[str]) -> bool | None:
    """Returns whether any expected candidate appears among the options or the pick.

    Args:
        brief: The brief as a dict.
        expect_any: Lowercase names that a good answer is likely to mention.

    Returns:
        True or False, or None when the topic has no expectation.
    """
    if not expect_any:
        return None
    names = [str(option.get("name", "")).lower() for option in brief.get("options") or []]
    names.append(str((brief.get("recommendation") or {}).get("choice", "")).lower())
    return any(expected.lower() in name for expected in expect_any for name in names)


def source_domains(sources: list[dict[str, Any]]) -> list[str]:
    """Returns the distinct hosts of the sources, without a leading www."""
    hosts = {urlparse(source.get("url", "")).netloc.lower().removeprefix("www.") for source in sources}
    return sorted(host for host in hosts if host)


def extract_claims(brief: dict[str, Any]) -> list[dict[str, Any]]:
    """Collects the checkable statements of a brief, capped for judging cost.

    Comparison assessments come first, then option strengths and weaknesses, and
    the recommendation rationale always keeps the last slot.

    Args:
        brief: The brief as a dict.

    Returns:
        Claims with a 1-based id, their text and the source numbers they cite.
    """
    texts: list[tuple[str, list[int]]] = []
    for entry in brief.get("comparison") or []:
        texts.append(
            (f"{entry.get('option')} on {entry.get('criterion')}: {entry.get('assessment')}", entry.get("source_ids") or [])
        )
    for option in brief.get("options") or []:
        for item in option.get("strengths") or []:
            texts.append((f"{option.get('name')} strength: {item}", option.get("source_ids") or []))
        for item in option.get("weaknesses") or []:
            texts.append((f"{option.get('name')} weakness: {item}", option.get("source_ids") or []))
    texts = texts[: MAX_CLAIMS - 1]
    recommendation = brief.get("recommendation") or {}
    if recommendation.get("rationale"):
        texts.append((f"Recommendation ({recommendation.get('choice')}): {recommendation['rationale']}", []))
    return [{"id": index, "text": text, "cited": cited} for index, (text, cited) in enumerate(texts, 1)]


def format_sources(sources: list[dict[str, Any]]) -> str:
    """Formats numbered sources with their snippets for the judge prompt."""
    lines: list[str] = []
    for index, source in enumerate(sources, 1):
        lines.append(f"[{index}] {source.get('title', '')} ({source.get('source', '')})")
        lines.append(f"Content: {source.get('content', '')}")
        lines.append("")
    return "\n".join(lines)


def format_judge_prompt(claims: list[dict[str, Any]], sources_text: str) -> str:
    """Builds the user prompt asking the judge to verify every claim."""
    numbered = "\n".join(
        f"{claim['id']}. {claim['text']}"
        + (f" (cites {''.join(f'[{i}]' for i in claim['cited'])})" if claim["cited"] else "")
        for claim in claims
    )
    return (
        f"Sources:\n{sources_text}\n\nClaims to check:\n{numbered}\n\n"
        f'Respond with JSON: {{"verdicts": [...]}} containing exactly {len(claims)} entries, '
        "each one of supported, partial or unsupported, in claim order."
    )


def normalize_verdicts(raw: list[Any], count: int) -> list[str]:
    """Maps free-form judge verdicts to supported, partial, unsupported or unjudged.

    Args:
        raw: The verdicts returned by the judge.
        count: The number of claims that were sent.

    Returns:
        Exactly `count` verdicts. Missing ones are 'unjudged' and unrecognized
        ones count as 'unsupported'.
    """
    verdicts = [VERDICT_ALIASES.get(str(item).strip().lower(), "unsupported") for item in raw[:count]]
    return verdicts + ["unjudged"] * (count - len(verdicts))


def groundedness_score(verdicts: list[str]) -> float | None:
    """Returns the mean support score over the judged claims, or None when none were judged."""
    scores = [VERDICT_SCORES[verdict] for verdict in verdicts if verdict in VERDICT_SCORES]
    return mean(scores)


def unsupported_examples(claims: list[dict[str, Any]], verdicts: list[str], limit: int = 3) -> list[str]:
    """Returns up to `limit` claim texts that the judge marked unsupported."""
    return [claim["text"] for claim, verdict in zip(claims, verdicts) if verdict == "unsupported"][:limit]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates per-topic results into the headline metrics.

    Args:
        results: One result dict per evaluated topic.

    Returns:
        Success, structure, citation, topic, groundedness, latency and token
        metrics, plus node timings, confidence distribution and failure codes.
    """
    ok = [r for r in results if r["ok"]]
    latencies = [r["latency_seconds"] for r in ok]
    node_totals: dict[str, list[float]] = {}
    for result in ok:
        for node, seconds in result["node_seconds"].items():
            node_totals.setdefault(node, []).append(seconds)
    confidence: dict[str, int] = {}
    for result in ok:
        confidence[result["confidence"]] = confidence.get(result["confidence"], 0) + 1
    failures: dict[str, int] = {}
    for result in results:
        if not result["ok"]:
            failures[result["error_code"]] = failures.get(result["error_code"], 0) + 1
    expected = [r["topic_covered"] for r in ok if r["topic_covered"] is not None]
    return {
        "topics": len(results),
        "succeeded": len(ok),
        "success_rate": len(ok) / len(results) if results else None,
        "structure_ok_rate": mean([1.0 if not r["structure_issues"] else 0.0 for r in ok]),
        "citation_coverage": mean([r["citation"]["coverage"] for r in ok]),
        "topic_coverage_rate": mean([1.0 if covered else 0.0 for covered in expected]),
        "groundedness": mean([r["groundedness"] for r in ok if r["groundedness"] is not None]),
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "tokens_mean_total": mean([r["tokens"]["total"] for r in ok]),
        "tokens_mean_input": mean([r["tokens"]["input"] for r in ok]),
        "tokens_mean_output": mean([r["tokens"]["output"] for r in ok]),
        "sources_mean": mean([r["sources_count"] for r in ok]),
        "domains_mean": mean([len(r["domains"]) for r in ok]),
        "node_seconds_mean": {node: mean(values) for node, values in sorted(node_totals.items())},
        "confidence": confidence,
        "failures": failures,
    }


def regressions(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Compares a run with a baseline and lists metrics that got worse.

    Args:
        current: The summary of the new run.
        baseline: The summary of the saved baseline.

    Returns:
        Human-readable regression messages, empty when nothing got worse.
    """
    found: list[str] = []
    for metric in HIGHER_IS_BETTER:
        new, old = current.get(metric), baseline.get(metric)
        if new is not None and old is not None and new < old - REGRESSION_TOLERANCE:
            found.append(f"{metric} dropped from {old:.2f} to {new:.2f}")
    new_p95, old_p95 = current.get("latency_p95_seconds"), baseline.get("latency_p95_seconds")
    if new_p95 is not None and old_p95 is not None and new_p95 > old_p95 * LATENCY_REGRESSION_RATIO:
        found.append(f"latency_p95_seconds rose from {old_p95:.1f} to {new_p95:.1f}")
    return found


def _fmt(value: float | None, digits: int = 2) -> str:
    """Formats an optional number for the report."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    """Renders the headline metrics as a markdown table for the console and the README.

    Args:
        summary: The aggregated metrics.
        meta: Run metadata such as models and the judge.

    Returns:
        A markdown report.
    """
    rows = [
        ("Topics evaluated", f"{summary['succeeded']}/{summary['topics']} succeeded ({_fmt(summary['success_rate'])})"),
        ("Valid structure", _fmt(summary["structure_ok_rate"])),
        ("Citation coverage", _fmt(summary["citation_coverage"])),
        ("Expected candidates mentioned", _fmt(summary["topic_coverage_rate"])),
        ("Groundedness (LLM judge)", _fmt(summary["groundedness"])),
        ("Latency p50 / p95 (s)", f"{_fmt(summary['latency_p50_seconds'], 1)} / {_fmt(summary['latency_p95_seconds'], 1)}"),
        ("Tokens per run (in / out / total)", f"{_fmt(summary['tokens_mean_input'], 0)} / {_fmt(summary['tokens_mean_output'], 0)} / {_fmt(summary['tokens_mean_total'], 0)}"),
        ("Sources / distinct domains", f"{_fmt(summary['sources_mean'], 1)} / {_fmt(summary['domains_mean'], 1)}"),
        ("Confidence", json.dumps(summary["confidence"], sort_keys=True)),
        ("Failures", json.dumps(summary["failures"], sort_keys=True) if summary["failures"] else "none"),
    ]
    lines = [
        f"## Research eval ({meta.get('timestamp', '')})",
        f"Models: writer `{meta.get('writer_model')}`, chat `{meta.get('chat_model')}`, judge `{meta.get('judge_model') or 'off'}`",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value} |" for name, value in rows]
    node_rows = [f"| {node} | {_fmt(seconds, 1)} |" for node, seconds in summary["node_seconds_mean"].items()]
    if node_rows:
        lines += ["", "| Node | Mean seconds |", "|---|---|", *node_rows]
    return "\n".join(lines)