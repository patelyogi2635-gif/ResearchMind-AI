import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.core.llm import MAX_RATE_LIMIT_WAIT_SECONDS, get_llm, rate_limit_delay
from app.core.prompts import PLANNER_SYSTEM, PLANNER_USER
from app.core.runtime import get_runtime
from app.schemas.research import PlanOutput
from app.schemas.state import ResearchState

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 4
MAX_PLANNER_ATTEMPTS = 2
ERROR_LOG_CHARS = 300
PLANNER_JSON_INSTRUCTIONS = (
    'Respond with only a JSON object: {"sub_questions": ["question 1", "question 2", "question 3", "question 4"]}'
)


async def decompose_query(state: ResearchState, config: RunnableConfig) -> dict:
    """Graph node that breaks the research topic into focused sub-questions.

    Uses JSON mode and validates the reply against PlanOutput. Tool calling was
    unreliable here: the model sometimes answered in plain text and the provider
    rejected the call, which silently reduced research to a single search. The call
    is retried once, and a provider rate limit is waited out. If no plan is
    produced, the raw topic becomes the single sub-question so the pipeline can
    still attempt a search.

    Args:
        state: Current pipeline state containing the user's query.
        config: Run config carrying the shared runtime.

    Returns:
        A partial state update with the `sub_questions` field.
    """
    runtime = get_runtime(config)
    query = state["query"]
    logger.info("planner.start query=%r", query)

    planner = get_llm(temperature=0.3).with_structured_output(PlanOutput, method="json_mode")
    messages = [
        SystemMessage(content=PLANNER_SYSTEM + "\n\n" + PLANNER_JSON_INSTRUCTIONS),
        HumanMessage(content=PLANNER_USER.format(query=query)),
    ]
    plan = None
    for attempt in range(1, MAX_PLANNER_ATTEMPTS + 1):
        try:
            async with runtime.llm_semaphore:
                plan = await planner.ainvoke(messages)
        except Exception as exc:
            delay = rate_limit_delay(exc)
            logger.warning(
                "planner.attempt_failed attempt=%d rate_limited=%s error=%s",
                attempt,
                delay is not None,
                str(exc)[:ERROR_LOG_CHARS],
            )
            if delay is not None and attempt < MAX_PLANNER_ATTEMPTS:
                await asyncio.sleep(min(delay, MAX_RATE_LIMIT_WAIT_SECONDS))
            continue
        if plan is not None:
            break

    sub_questions = (
        [q.strip() for q in plan.sub_questions if q.strip()][:MAX_SUB_QUESTIONS] if plan else []
    )
    if not sub_questions:
        logger.warning("planner.fallback reason=no_plan using=raw_query")
        sub_questions = [query]

    logger.info("planner.done count=%d", len(sub_questions))
    return {"sub_questions": sub_questions}