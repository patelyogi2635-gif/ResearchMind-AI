import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.errors import AppError
from app.core.llm import MAX_RATE_LIMIT_WAIT_SECONDS, get_router_llm, rate_limit_delay
from app.core.prompts import (
    DEFAULT_REFUSAL,
    FALLBACK_REPLIES,
    REFUSAL_MESSAGES,
    ROUTER_JSON_INSTRUCTIONS,
    ROUTER_SYSTEM,
    strip_delimiters,
    wrap_user_message,
)
from app.schemas.chat import ChatReply, Intent, RouteDecision
from app.schemas.context import Runtime
from app.schemas.memory import ConversationContext

logger = logging.getLogger(__name__)

MAX_ROUTER_ATTEMPTS = 3
MAX_REPLY_CHARS = 400
MAX_STANDALONE_CHARS = 500
ROUTER_BRIEF_CHARS = 1500
ERROR_LOG_CHARS = 300
FORBIDDEN_REPLY_MARKERS = ("http", "```", "<")
ROUTER_PROMPT = ROUTER_SYSTEM + "\n\n" + ROUTER_JSON_INSTRUCTIONS


def _render_context(context: ConversationContext | None) -> str:
    """Renders the conversation context block for the router prompt."""
    if context is None or (not context.turns and not context.brief_summary):
        return "<conversation>\nNo previous conversation.\n</conversation>"
    lines = ["<conversation>"]
    if context.turns:
        lines.append("Recent turns:")
        lines.extend(strip_delimiters(turn) for turn in context.turns)
    if context.brief_summary:
        lines.append("Latest research brief:")
        lines.append(strip_delimiters(context.brief_summary[:ROUTER_BRIEF_CHARS]))
    else:
        lines.append("No research brief exists yet.")
    lines.append("</conversation>")
    return "\n".join(lines)


def _apply_context_rules(decision: RouteDecision, context: ConversationContext | None) -> RouteDecision:
    """Enforces rules the model cannot be trusted to follow on its own.

    A follow-up without a stored brief is downgraded to unclear, and the
    standalone query is kept only for research and only when it has a sane length.
    """
    has_brief = bool(context and context.brief_summary)
    if decision.intent == Intent.FOLLOW_UP and not has_brief:
        logger.info("router.follow_up_without_brief downgraded=unclear")
        decision.intent = Intent.UNCLEAR
        decision.reply = ""
    standalone = decision.standalone_query.strip()
    keep = decision.intent == Intent.RESEARCH and 0 < len(standalone) <= MAX_STANDALONE_CHARS
    decision.standalone_query = standalone if keep else ""
    return decision


async def route_message(
    runtime: Runtime, message: str, context: ConversationContext | None = None
) -> RouteDecision:
    """Classifies a message with one small LLM call before any work runs.

    The router decides whether the message is a greeting, a research request, a
    follow-up about the previous brief, off topic, unsafe or unclear. It has no
    tools and returns constrained JSON, so a manipulated router can at worst
    choose a wrong route. Provider rate limits are waited out for the suggested
    time. If no valid decision is produced, the request is rejected instead of
    running unchecked.

    Args:
        runtime: Shared runtime holding the LLM concurrency limit.
        message: The raw user message.
        context: Recent turns and a summary of the latest brief, if any.

    Returns:
        The validated routing decision.

    Raises:
        AppError: With status 429 when the provider keeps rate limiting, or 503
            when no valid decision is produced.
    """
    router = get_router_llm().with_structured_output(RouteDecision, method="json_mode")
    prompt = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=f"{_render_context(context)}\n\n{wrap_user_message(message)}"),
    ]

    rate_limited = False
    for attempt in range(1, MAX_ROUTER_ATTEMPTS + 1):
        try:
            async with runtime.llm_semaphore:
                decision = await router.ainvoke(prompt)
        except Exception as exc:
            delay = rate_limit_delay(exc)
            rate_limited = delay is not None
            logger.warning(
                "router.attempt_failed attempt=%d rate_limited=%s error=%s",
                attempt,
                rate_limited,
                str(exc)[:ERROR_LOG_CHARS],
            )
            if delay is not None and attempt < MAX_ROUTER_ATTEMPTS:
                await asyncio.sleep(min(delay, MAX_RATE_LIMIT_WAIT_SECONDS))
            continue
        if decision is None:
            rate_limited = False
            continue
        decision = _apply_context_rules(decision, context)
        if decision.intent == Intent.UNSAFE:
            logger.warning(
                "router.blocked category=%s message_chars=%d",
                decision.unsafe_category.value,
                len(message),
            )
        else:
            logger.info("router.decided intent=%s", decision.intent.value)
        return decision

    if rate_limited:
        raise AppError(
            "The assistant is busy right now. Please try again in a few seconds.",
            status_code=429,
            code="llm_rate_limited",
        )
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
    """Builds the user-facing reply for a greeting, off-topic, unsafe or unclear decision.

    Refusals always use fixed templates, so they stay consistent and never reveal
    internal rules. Model-written replies are used only when they pass basic
    checks, otherwise a fixed fallback is used.

    Args:
        decision: A routing decision that is neither research nor follow-up.

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