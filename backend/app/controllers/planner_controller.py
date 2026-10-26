import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.llm import get_llm
from app.core.prompts import PLANNER_SYSTEM, PLANNER_USER
from app.schemas.research import PlanOutput
from app.schemas.state import ResearchState

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 4


async def decompose_query(state: ResearchState) -> dict:
    """Graph node that breaks the research topic into focused sub-questions.

    Uses structured output so the result is validated against PlanOutput. If the
    model call or parsing fails, the raw topic is used as the single sub-question
    so the pipeline can still attempt a search.

    Args:
        state: Current pipeline state containing the user's query.

    Returns:
        A partial state update with the `sub_questions` field.
    """
    query = state["query"]
    logger.info("planner.start query=%r", query)

    planner = get_llm(temperature=0.3).with_structured_output(PlanOutput)
    try:
        plan = await planner.ainvoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=PLANNER_USER.format(query=query)),
            ]
        )
    except Exception as exc:
        logger.warning("planner.llm_failed error=%s fallback=raw_query", exc)
        return {"sub_questions": [query]}

    sub_questions = (
        [q.strip() for q in plan.sub_questions if q.strip()][:MAX_SUB_QUESTIONS] if plan else []
    )
    if not sub_questions:
        logger.warning("planner.empty_plan fallback=raw_query")
        sub_questions = [query]

    logger.info("planner.done count=%d", len(sub_questions))
    return {"sub_questions": sub_questions}


def build_planner_graph():
    """Builds and compiles the planner graph: decompose_query -> END."""
    graph = StateGraph(ResearchState)
    graph.add_node("decompose_query", decompose_query)
    graph.set_entry_point("decompose_query")
    graph.add_edge("decompose_query", END)
    return graph.compile()


planner_graph = build_planner_graph()
