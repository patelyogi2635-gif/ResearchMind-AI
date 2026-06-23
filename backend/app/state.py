"""
state.py — the single shared state object for the entire research pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: LangGraph State
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In LangGraph, every node in every graph reads from and writes to
a single shared "state" object — a TypedDict (or Pydantic model).

Think of it like a baton in a relay race:
  Node 1 (Planner) receives state, adds sub_questions, passes it on.
  Node 2 (Researcher) receives state, reads sub_questions, adds search_results.
  Node 3 (Writer) receives state, reads everything, adds the final report.

The state is IMMUTABLE between nodes — each node returns a PARTIAL update,
and LangGraph merges it with the existing state using a "reducer" function.

For simple fields (str, dict), the default reducer is: last write wins.
For lists (like search_results), you usually want: append / accumulate.
That's what Annotated[list, operator.add] does below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY TypedDict over a dataclass or Pydantic model?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph's StateGraph requires TypedDict (or a subclass of it).
Pydantic models are supported but require extra setup.
TypedDict is the simplest and most idiomatic choice for LangGraph.
"""

import operator
from typing import Annotated, TypedDict


class SearchResult(TypedDict):
    """A single result from one MCP tool call."""
    sub_question: str          # which sub-question this answers
    source: str                # "tavily" or "github"
    title: str
    url: str
    content: str               # raw snippet / README excerpt
    relevance_score: float     # 0.0–1.0, added by scorer node


class ResearchState(TypedDict):
    """
    The complete state object for the research pipeline.

    Fields are populated progressively as the pipeline runs:

    START:    only `query` is set
    After Planner:   `sub_questions` is populated
    After Research:  `search_results` is populated
    After Writer:    `report` and `sources` are populated
    """

    # ── Input ──────────────────────────────────────────────
    query: str
    # "How does LangGraph compare to CrewAI for production use?"

    # ── Planner output ─────────────────────────────────────
    sub_questions: list[str]
    # ["What is LangGraph?", "What is CrewAI?", "How do they handle state?", ...]

    # ── Research output ────────────────────────────────────
    # Annotated with operator.add means: when a node returns
    # {"search_results": [new_item]}, LangGraph APPENDS to the list
    # rather than replacing it. Essential for parallel research loops.
    search_results: Annotated[list[SearchResult], operator.add]

    # ── Writer output ──────────────────────────────────────
    sources: list[SearchResult]   # deduplicated, scored, top-N
    report: str                   # final markdown report with citations


class PipelineStatus(TypedDict):
    """Lightweight status object used for SSE event streaming."""
    session_id: str
    phase: str          # "planning" | "researching" | "writing" | "done" | "error"
    message: str
    data: dict          # phase-specific payload (sub_questions, source URLs, etc.)