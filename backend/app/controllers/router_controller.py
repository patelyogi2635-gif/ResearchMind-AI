import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.errors import AppError
from app.core.llm import get_router_llm
from app.core.prompts import (
    DEFAULT_REFUSAL,
    FALLBACK_REPLIES,
    REFUSAL_MESSAGES,
    ROUTER_JSON_INSTRUCTIONS,
    ROUTER_SYSTEM,
)
from app.schemas.chat import ChatReply, Intent, RouteDecision
from app.schemas.context import Runtime

logger = logging.getLogger(__name__)

MAX_ROUTER_ATTEMPTS = 2
MAX_REPLY_CHARS = 400
ERROR_LOG_CHARS = 300
FORBIDDEN_REPLY_MARKERS = ("http", "```", "<")
DELIMITER_PATTERN = re.compile(r"</?user_message>", re.IGNORECASE)
ROUTE_SCHEMA_JSON = json.dumps(RouteDecision.model_json_schema(), separators=(",", ":"))


def _wrap_user_message(message: str) -> str:
    """Wraps the user message in delimiters, removing any delimiter it contains."""
    cleaned = DELIMITER_PATTERN.sub("", message)
    return f"<user_message>\n{cleaned}\n</user_message>"


async def route_message(runtime: Runtime, message: str) -> RouteDecision:
    """Classifies a message with one small LLM call before any research runs.

    The router decides whether the message is a greeting, a research request, off
    topic, unsafe or unclear. It has no tools and returns constrained JSON, so a
    manipulated router can at worst choose a wrong route. If it cannot produce a
    valid decision, the request is rejected instead of running unchecked.

    Args:
        runtime: Shared runtime holding the LLM concurrency limit.
        message: The raw user message.

    Returns:
        The validated routing decision.

    Raises:
        AppError: With status 503 when no valid decision is produced.
    """
    router = get_router_llm().with_structured_output(RouteDecision, method="json_mode")
    prompt = [
        SystemMessage(
            content=ROUTER_SYSTEM + "\n\n" + ROUTER_JSON_INSTRUCTIONS.format(schema=ROUTE_SCHEMA_JSON)
        ),
        HumanMessage(content=_wrap_user_message(message)),
    ]

    for attempt in range(1, MAX_ROUTER_ATTEMPTS + 1):
        try:
            async with runtime.llm_semaphore:
                decision = await router.ainvoke(prompt)
        except Exception as exc:
            logger.warning(
                "router.attempt_failed attempt=%d error=%s", attempt, str(exc)[:ERROR_LOG_CHARS]
            )
            continue
        if decision is None:
            continue
        if decision.intent == Intent.UNSAFE:
            logger.warning(
                "router.blocked category=%s message_chars=%d",
                decision.unsafe_category.value,
                len(message),
            )
        else:
            logger.info("router.decided intent=%s", decision.intent.value)
        return decision

    raise AppError(
        "I couldn't process that message right now. Please try again.",
        status_code=503,
        code="router_unavailable",
    )


def _clean_reply(text: str) -> str:
    """Returns the model's reply only if it is short and free of links or markup."""
    reply = text.strip()
    if not reply or len(reply) > MAX_REPLY_CHARS:
        return ""
    if any(marker in reply for marker in FORBIDDEN_REPLY_MARKERS):
        return ""
    return reply


def build_reply(decision: RouteDecision) -> ChatReply:
    """Builds the user-facing reply for a non-research decision.

    Refusals always use fixed templates, so they stay consistent and never reveal
    internal rules. Model-written replies are used only when they pass basic
    checks, otherwise a fixed fallback is used.

    Args:
        decision: A routing decision whose intent is not research.

    Returns:
        The reply text with its intent.
    """
    if decision.intent == Intent.UNSAFE:
        text = REFUSAL_MESSAGES.get(decision.unsafe_category.value, DEFAULT_REFUSAL)
    else:
        text = _clean_reply(decision.reply) or FALLBACK_REPLIES.get(
            decision.intent.value, FALLBACK_REPLIES["off_topic"]
        )
    return ChatReply(intent=decision.intent, text=text)