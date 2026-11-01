import asyncio
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import MAX_RATE_LIMIT_WAIT_SECONDS, get_chat_llm, rate_limit_delay
from app.core.prompts import FOLLOW_UP_FALLBACK, FOLLOW_UP_SYSTEM, strip_delimiters, wrap_user_message
from app.core.runtime import EmitFn
from app.schemas.context import Runtime
from app.schemas.memory import ConversationContext

logger = logging.getLogger(__name__)

STREAM_ATTEMPTS = 2


async def _stream_once(
    runtime: Runtime, llm, messages: list, emit: EmitFn, target: str, parts: list[str]
) -> str:
    """Streams one completion, emitting small coalesced chunks as they arrive.

    Chunks are flushed when enough characters have accumulated or enough time has
    passed since the last flush, so the first words appear quickly without one
    event per token.

    Args:
        runtime: Shared runtime holding the LLM concurrency limit and settings.
        llm: The chat model.
        messages: The prompt messages.
        emit: Async callback receiving (event, data).
        target: Where the client should render the text, "answer" or "summary".
        parts: Collects every received text piece, shared with the caller.

    Returns:
        The complete generated text.
    """
    settings = runtime.settings
    buffer = ""
    last_flush = time.monotonic()
    async with runtime.llm_semaphore:
        async for chunk in llm.astream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                continue
            parts.append(text)
            buffer += text
            now = time.monotonic()
            if len(buffer) >= settings.stream_chunk_chars or now - last_flush >= settings.stream_flush_seconds:
                await emit("token", {"text": buffer, "target": target})
                buffer = ""
                last_flush = now
    if buffer:
        await emit("token", {"text": buffer, "target": target})
    return "".join(parts)


async def _invoke_once(runtime: Runtime, llm, messages: list, emit: EmitFn, target: str) -> str:
    """Generates a completion without streaming and emits it as a single chunk."""
    async with runtime.llm_semaphore:
        message = await llm.ainvoke(messages)
    text = message.content if isinstance(message.content, str) else ""
    if text:
        await emit("token", {"text": text, "target": target})
    return text


async def stream_completion(
    runtime: Runtime, llm, messages: list, emit: EmitFn, target: str = "answer"
) -> str:
    """Generates text and streams it to the client through `emit`.

    Streaming can be switched off with STREAMING_ENABLED, in which case the text
    is sent as one chunk, so clients handle both modes identically. A provider
    rate limit is waited out and retried only while nothing has been shown yet.

    Args:
        runtime: Shared runtime.
        llm: The chat model.
        messages: The prompt messages.
        emit: Async callback receiving (event, data).
        target: Where the client should render the text.

    Returns:
        The complete generated text.

    Raises:
        Exception: Any model error that is not a retryable rate limit.
    """
    for attempt in range(1, STREAM_ATTEMPTS + 1):
        parts: list[str] = []
        try:
            if runtime.settings.streaming_enabled:
                return await _stream_once(runtime, llm, messages, emit, target, parts)
            return await _invoke_once(runtime, llm, messages, emit, target)
        except Exception as exc:
            delay = rate_limit_delay(exc)
            if delay is not None and not parts and attempt < STREAM_ATTEMPTS:
                logger.warning("stream.rate_limited attempt=%d wait=%.1fs", attempt, delay)
                await asyncio.sleep(min(delay, MAX_RATE_LIMIT_WAIT_SECONDS))
                continue
            raise
    return ""


def _follow_up_prompt(question: str, context: ConversationContext | None) -> str:
    """Builds the follow-up prompt from the stored brief, recent turns and the question."""
    brief = strip_delimiters(context.brief_summary) if context and context.brief_summary else "No brief."
    turns = "\n".join(strip_delimiters(turn) for turn in (context.turns if context else []))
    return f"<brief>\n{brief}\n</brief>\n<recent_turns>\n{turns or 'None.'}\n</recent_turns>\n{wrap_user_message(question)}"


async def answer_follow_up(
    runtime: Runtime, question: str, context: ConversationContext | None, emit: EmitFn
) -> str:
    """Answers a follow-up question from the stored brief, streaming the answer.

    No search runs: the answer must come from the brief and the recent turns.

    Args:
        runtime: Shared runtime.
        question: The user's follow-up question.
        context: Recent turns and a compact summary of the latest brief.
        emit: Async callback receiving (event, data).

    Returns:
        The answer text, or a fixed fallback when the model returns nothing.
    """
    messages = [
        SystemMessage(content=FOLLOW_UP_SYSTEM),
        HumanMessage(content=_follow_up_prompt(question, context)),
    ]
    text = (await stream_completion(runtime, get_chat_llm(), messages, emit, target="answer")).strip()
    return text or FOLLOW_UP_FALLBACK