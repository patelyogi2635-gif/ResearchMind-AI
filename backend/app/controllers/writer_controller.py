import json
import logging
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.core.errors import AppError
from app.core.llm import get_llm
from app.core.prompts import WRITER_JSON_INSTRUCTIONS, WRITER_SYSTEM, WRITER_USER
from app.core.runtime import get_runtime
from app.schemas.research import Confidence, RDBrief
from app.schemas.state import ResearchState, SearchResult

logger = logging.getLogger(__name__)

MAX_WRITER_WEB_SOURCES = 4
MAX_WRITER_GITHUB_SOURCES = 2
MAX_WRITER_SOURCES = 6
MAX_BRIEF_ATTEMPTS = 3
WRITER_MAX_TOKENS = 8000
WRITER_TEMPERATURE = 0.2
ERROR_LOG_CHARS = 300
MIN_INDEPENDENT_DOMAINS = 2

CONFIDENCE_RANK = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
BRIEF_SCHEMA_JSON = json.dumps(RDBrief.model_json_schema(), separators=(",", ":"))


def _format_sources_for_prompt(sources: list[SearchResult]) -> str:
    """Formats sources as a numbered list the model can cite by number."""
    if not sources:
        return "No sources available."
    lines: list[str] = []
    for index, source in enumerate(sources, 1):
        lines.append(f"[{index}] {source['title']} ({source['source']})")
        lines.append(f"URL: {source['url']}")
        lines.append(f"Content: {source['content']}")
        lines.append("")
    return "\n".join(lines)


def _clean_ids(ids: list[int], source_count: int) -> list[int]:
    """Drops citation numbers that do not match a real source, sorted and unique."""
    return sorted({i for i in ids if 1 <= i <= source_count})


def sanitize_citations(brief: RDBrief, source_count: int) -> RDBrief:
    """Removes citation numbers the model invented.

    Args:
        brief: The generated brief.
        source_count: Number of sources that were actually provided.

    Returns:
        The same brief with only valid source numbers left on options and
        comparison entries.
    """
    for option in brief.options:
        option.source_ids = _clean_ids(option.source_ids, source_count)
    for entry in brief.comparison:
        entry.source_ids = _clean_ids(entry.source_ids, source_count)
    return brief


def _domain(url: str) -> str:
    """Returns the host of a URL without a leading www."""
    return urlparse(url).netloc.lower().removeprefix("www.")


def _names_match(left: str, right: str) -> bool:
    """Case-insensitive containment match between two option names."""
    a, b = left.strip().lower(), right.strip().lower()
    return bool(a) and bool(b) and (a in b or b in a)


def _supporting_source_ids(brief: RDBrief) -> set[int]:
    """Collects the source numbers that back the recommended option.

    Falls back to every cited source when the recommendation does not name a
    single evaluated option.
    """
    choice = brief.recommendation.choice
    ids: set[int] = set()
    for option in brief.options:
        if _names_match(option.name, choice):
            ids.update(option.source_ids)
    for entry in brief.comparison:
        if _names_match(entry.option, choice):
            ids.update(entry.source_ids)
    if ids:
        return ids
    for option in brief.options:
        ids.update(option.source_ids)
    for entry in brief.comparison:
        ids.update(entry.source_ids)
    return ids


def cap_confidence(brief: RDBrief, sources: list[SearchResult]) -> RDBrief:
    """Caps the recommendation confidence by the number of independent domains.

    One source domain supports at most medium confidence, and no supporting
    source caps it at low. A note is added to the limitations when a cap applies.

    Args:
        brief: The generated brief with sanitized citations.
        sources: The numbered sources the brief cites.

    Returns:
        The same brief, with confidence lowered when the evidence is too thin.
    """
    domains = {
        _domain(sources[i - 1]["url"])
        for i in _supporting_source_ids(brief)
        if 1 <= i <= len(sources)
    }
    if not domains:
        cap = Confidence.LOW
    elif len(domains) < MIN_INDEPENDENT_DOMAINS:
        cap = Confidence.MEDIUM
    else:
        return brief

    recommendation = brief.recommendation
    if CONFIDENCE_RANK[recommendation.confidence] > CONFIDENCE_RANK[cap]:
        recommendation.confidence = cap
        brief.limitations.append(
            f"Confidence capped at {cap.value}: the recommendation is backed by "
            f"{len(domains)} independent source domain(s)."
        )
    return brief


def _cite(ids: list[int]) -> str:
    """Renders citation numbers as ' [1][2]' or an empty string."""
    return " " + "".join(f"[{i}]" for i in ids) if ids else ""


def _clean_title(title: str) -> str:
    """Replaces square brackets so a title cannot break markdown link syntax."""
    return " ".join(title.replace("[", "(").replace("]", ")").split())


def render_brief_markdown(brief: RDBrief, sources: list[SearchResult]) -> str:
    """Renders a brief as markdown compatible with the frontend renderer.

    The frontend supports only `##`/`###` headings, `**bold**`, `* ` bullets and
    links, so tables and numbered lists are intentionally avoided.

    Args:
        brief: The structured R&D brief.
        sources: The numbered sources the brief cites.

    Returns:
        The brief as a markdown string.
    """
    lines: list[str] = [
        f"## {brief.title}",
        "",
        "## Executive Summary",
        "",
        brief.executive_summary,
        "",
        "## Problem Statement",
        "",
        brief.problem_statement,
        "",
        "## Landscape",
        "",
        brief.landscape,
        "",
        "## Options Evaluated",
        "",
    ]

    for option in brief.options:
        lines += [
            f"### {option.name}",
            f"**Maturity:** {option.maturity.value}{_cite(option.source_ids)}",
            option.summary,
        ]
        lines += [f"* **Strength:** {item}" for item in option.strengths]
        lines += [f"* **Weakness:** {item}" for item in option.weaknesses]
        lines.append("")

    if brief.comparison:
        grouped: dict[str, list] = {}
        for entry in brief.comparison:
            grouped.setdefault(entry.criterion, []).append(entry)
        lines += ["## Comparison", ""]
        for criterion, entries in grouped.items():
            lines += [f"### {criterion}"]
            lines += [
                f"* **{entry.option}:** {entry.assessment}{_cite(entry.source_ids)}"
                for entry in entries
            ]
            lines.append("")

    if brief.risks:
        lines += ["## Risks", ""]
        for risk in brief.risks:
            mitigation = f" Mitigation: {risk.mitigation}" if risk.mitigation else ""
            lines.append(f"* **{risk.severity.value.upper()}:** {risk.description}{mitigation}")
        lines.append("")

    recommendation = brief.recommendation
    lines += [
        "## Recommendation",
        "",
        f"**Recommended:** {recommendation.choice} (confidence: {recommendation.confidence.value})",
        recommendation.rationale,
    ]
    if recommendation.next_steps:
        lines.append("**Next steps:**")
        lines += [f"* {step}" for step in recommendation.next_steps]
    lines.append("")

    if brief.poc_plan:
        lines += ["## Proof-of-Concept Plan", ""]
        lines += [f"* Step {number}: {step}" for number, step in enumerate(brief.poc_plan, 1)]
        lines.append("")

    if brief.limitations:
        lines += ["## Limitations and Unverified Items", ""]
        lines += [f"* {item}" for item in brief.limitations]
        lines.append("")

    if sources:
        lines += ["## Sources", ""]
        lines += [
            f"* [{index}] [{_clean_title(source['title'])}]({source['url']})"
            for index, source in enumerate(sources, 1)
        ]

    return "\n".join(lines).strip()


async def select_sources(state: ResearchState, config: RunnableConfig) -> dict:
    """Graph node that picks the best sources for the writer prompt.

    Args:
        state: Current pipeline state containing ranked `sources`.
        config: Run config (unused, kept for a uniform node signature).

    Returns:
        A partial state update with the selected `sources`.
    """
    all_sources = state.get("sources", [])
    web = [s for s in all_sources if s["source"] == "tavily"][:MAX_WRITER_WEB_SOURCES]
    github = [s for s in all_sources if s["source"] == "github"][:MAX_WRITER_GITHUB_SOURCES]
    selected = sorted(web + github, key=lambda s: s["relevance_score"], reverse=True)
    selected = selected[:MAX_WRITER_SOURCES]
    logger.info("writer.sources_selected count=%d", len(selected))
    return {"sources": selected}


async def synthesize_brief(state: ResearchState, config: RunnableConfig) -> dict:
    """Graph node that generates the structured R&D brief and its markdown form.

    Args:
        state: Current pipeline state with `query`, `sub_questions` and `sources`.
        config: Run config carrying the shared runtime.

    Returns:
        A partial state update with `brief` and `report`.

    Raises:
        AppError: If no valid structured brief is produced after all attempts.
    """
    runtime = get_runtime(config)
    query = state["query"]
    sources = state.get("sources", [])
    logger.info("writer.start sources=%d", len(sources))

    messages = [
        SystemMessage(
            content=WRITER_SYSTEM + "\n\n" + WRITER_JSON_INSTRUCTIONS.format(schema=BRIEF_SCHEMA_JSON)
        ),
        HumanMessage(
            content=WRITER_USER.format(
                query=query,
                sub_questions_text="\n".join(f"- {q}" for q in state.get("sub_questions", [])),
                sources_text=_format_sources_for_prompt(sources),
            )
        ),
    ]
    writer = get_llm(
        temperature=WRITER_TEMPERATURE, max_tokens=WRITER_MAX_TOKENS
    ).with_structured_output(RDBrief, method="json_mode")

    brief: RDBrief | None = None
    for attempt in range(1, MAX_BRIEF_ATTEMPTS + 1):
        try:
            async with runtime.llm_semaphore:
                brief = await writer.ainvoke(messages)
        except Exception as exc:
            logger.warning(
                "writer.attempt_failed attempt=%d error=%s", attempt, str(exc)[:ERROR_LOG_CHARS]
            )
            continue
        if brief is not None:
            break

    if brief is None:
        raise AppError(
            "Failed to generate a structured R&D brief",
            status_code=502,
            code="brief_generation_failed",
        )

    brief = cap_confidence(sanitize_citations(brief, len(sources)), sources)
    report = render_brief_markdown(brief, sources)
    logger.info(
        "writer.done options=%d confidence=%s report_chars=%d",
        len(brief.options),
        brief.recommendation.confidence.value,
        len(report),
    )
    return {"brief": brief, "report": report}