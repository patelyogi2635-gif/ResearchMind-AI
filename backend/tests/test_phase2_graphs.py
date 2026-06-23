"""
tests/test_phase2_graphs.py — integration tests for all three graphs.

Run with:
  cd backend/
  pytest tests/test_phase2_graphs.py -v -s

These make real API calls (Groq + MCP tools). Takes ~30-60 seconds.
"""

import pytest
from app.state import ResearchState
from app.graphs.planner import planner_graph
from app.graphs.research import research_graph
from app.graphs.writer import writer_graph
from app.services.pipeline import run_research_pipeline

TEST_QUERY = "What is LangGraph and how does it differ from LangChain?"


def make_initial_state(query: str = TEST_QUERY) -> ResearchState:
    return {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "report": "",
    }


# ── Test 1: Planner graph ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_graph():
    state = make_initial_state()
    result = await planner_graph.ainvoke(state)

    print(f"\n✓ Planner sub-questions:")
    for i, q in enumerate(result["sub_questions"], 1):
        print(f"  {i}. {q}")

    assert len(result["sub_questions"]) >= 1
    assert all(isinstance(q, str) for q in result["sub_questions"])
    assert result["query"] == TEST_QUERY  # query unchanged


# ── Test 2: Research graph ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_research_graph():
    # First get sub_questions from planner
    state = make_initial_state()
    state = await planner_graph.ainvoke(state)

    # Then run research
    state = await research_graph.ainvoke(state)

    print(f"\n✓ Research results:")
    print(f"  Raw results: {len(state['search_results'])}")
    print(f"  Scored sources: {len(state['sources'])}")
    for s in state["sources"][:3]:
        print(f"  [{s['source']}] {s['title'][:60]} (score: {s['relevance_score']:.2f})")

    assert len(state["sources"]) > 0
    assert all("url" in s for s in state["sources"])
    assert all("relevance_score" in s for s in state["sources"])


# ── Test 3: Writer graph ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_writer_graph():
    state = make_initial_state()
    state = await planner_graph.ainvoke(state)
    state = await research_graph.ainvoke(state)
    state = await writer_graph.ainvoke(state)

    print(f"\n✓ Report generated: {len(state['report'])} chars")
    print(f"\n--- REPORT PREVIEW (first 500 chars) ---")
    print(state["report"][:500])
    print("---")

    assert len(state["report"]) > 100
    assert state["query"] in state["report"] or len(state["report"]) > 200


# ── Test 4: Full pipeline ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_pipeline():
    state = await run_research_pipeline(TEST_QUERY)

    print(f"\n✓ Full pipeline complete:")
    print(f"  Sub-questions: {state['sub_questions']}")
    print(f"  Sources: {len(state['sources'])}")
    print(f"  Report: {len(state['report'])} chars")

    assert state["sub_questions"]
    assert state["sources"]
    assert len(state["report"]) > 200