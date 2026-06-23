"""
planner.py — Graph 1: Query decomposition.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: What is a LangGraph node?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A node is just a Python function that:
  1. Receives the current ResearchState as input
  2. Does something (calls LLM, calls tool, transforms data)
  3. Returns a DICT with only the fields it wants to update

LangGraph merges that dict with the existing state.
You never return the full state — only what changed.

Example:
  def my_node(state: ResearchState) -> dict:
      result = do_something(state["query"])
      return {"sub_questions": result}   # only the new field

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: What is a LangGraph graph?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A StateGraph wires nodes together with edges:
  - add_node(name, function)  → registers a node
  - add_edge(a, b)            → b always runs after a
  - add_conditional_edges()   → which node runs next depends on state
  - set_entry_point(name)     → first node to run
  - compile()                 → returns a runnable graph

The compiled graph exposes:
  - graph.invoke(state)         → runs synchronously, returns final state
  - graph.ainvoke(state)        → async version
  - graph.astream_events(state) → async, yields events as nodes fire
                                   (used in Phase 3 for SSE streaming)
"""

import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.state import ResearchState
from app.llm import get_llm
from app.graphs.prompts import PLANNER_SYSTEM, PLANNER_USER

logger = logging.getLogger(__name__)


# ── Node function ─────────────────────────────────────────────────────

async def decompose_query(state: ResearchState) -> dict:
    """
    Node: takes state.query, returns state.sub_questions.

    Uses Groq LLM with a strict JSON prompt to break the query
    into exactly 4 focused sub-questions.

    Why JSON output?
    Structured output = reliable parsing. The LLM is instructed to
    return ONLY a JSON object, which we parse directly. No regex needed.
    If parsing fails, we fall back to splitting the query into parts.
    """
    query = state["query"]
    logger.info("[Planner] Decomposing query: %s", query)

    llm = get_llm(temperature=0.3)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=PLANNER_USER.format(query=query)),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    logger.debug("[Planner] Raw LLM response: %s", raw)

    # Parse JSON response
    try:
        # Strip markdown fences if LLM added them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        sub_questions = data["sub_questions"]

        # Validate we got a list of strings
        if not isinstance(sub_questions, list) or len(sub_questions) == 0:
            raise ValueError("sub_questions must be a non-empty list")

        # Enforce max 4 questions to control API costs
        sub_questions = sub_questions[:4]

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("[Planner] JSON parse failed (%s), using fallback", e)
        # Fallback: treat the whole query as one question
        sub_questions = [query]

    logger.info("[Planner] Generated %d sub-questions: %s", len(sub_questions), sub_questions)

    # Return ONLY the fields we're updating
    return {"sub_questions": sub_questions}


# ── Graph builder ─────────────────────────────────────────────────────

def build_planner_graph() -> StateGraph:
    """
    Builds and compiles the Planner graph.

    Graph structure:
      START → decompose_query → END

    It's intentionally simple — one node, one job.
    Complexity lives in the Research graph.

    Returns a compiled graph ready to call with .ainvoke(state).
    """
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("decompose_query", decompose_query)

    # Wire edges
    graph.set_entry_point("decompose_query")
    graph.add_edge("decompose_query", END)

    compiled = graph.compile()
    logger.info("[Planner] Graph compiled successfully")
    return compiled


# ── Module-level compiled instance ────────────────────────────────────
# Build once at import time, reuse across all requests.
# This is the object the pipeline service will call.
planner_graph = build_planner_graph()