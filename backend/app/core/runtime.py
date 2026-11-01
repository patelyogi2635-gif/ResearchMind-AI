import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import Request
from langchain_core.runnables import RunnableConfig

from app.core.config import Settings
from app.core.errors import AppError
from app.core.mongo import close_mongo
from app.schemas.context import Runtime

logger = logging.getLogger(__name__)

CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


def create_runtime(settings: Settings) -> Runtime:
    """Creates the shared resources that live for the whole process.

    The HTTP client is created once so every request reuses pooled connections.
    The semaphores bound concurrent external calls and pipelines across all users.

    Args:
        settings: Application settings.

    Returns:
        A Runtime holding the HTTP client, concurrency limits and the job registry.
    """
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            settings.http_timeout_seconds, connect=settings.http_connect_timeout_seconds
        ),
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive_connections,
        ),
    )
    return Runtime(
        settings=settings,
        http=client,
        search_semaphore=asyncio.Semaphore(settings.max_concurrent_searches),
        llm_semaphore=asyncio.Semaphore(settings.max_concurrent_llm_calls),
        pipeline_semaphore=asyncio.Semaphore(settings.max_concurrent_pipelines),
    )


async def close_runtime(runtime: Runtime) -> None:
    """Stops running jobs, then closes MongoDB and the HTTP client in reverse order.

    Jobs are cancelled first so they can still save their outcome before the
    database connection closes.

    Args:
        runtime: The runtime created at startup.
    """
    tasks = [
        job.task
        for job in runtime.jobs.values()
        if job.task is not None and not job.task.done()
    ]
    if tasks:
        logger.info("shutdown.jobs cancelling=%d", len(tasks))
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await close_mongo(runtime)
    await runtime.http.aclose()
    logger.info("shutdown.http_client closed")


def get_runtime(config: RunnableConfig) -> Runtime:
    """Returns the runtime passed to a graph run through its config.

    Args:
        config: The LangGraph run config.

    Returns:
        The shared Runtime.

    Raises:
        AppError: If the graph was invoked without a runtime.
    """
    runtime = (config.get("configurable") or {}).get("runtime")
    if runtime is None:
        raise AppError("Runtime is not available", status_code=500, code="runtime_missing")
    return runtime


def get_app_runtime(request: Request) -> Runtime:
    """FastAPI dependency that returns the runtime created at startup.

    Args:
        request: The incoming request.

    Returns:
        The shared Runtime.

    Raises:
        AppError: If the application has not finished starting.
    """
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise AppError("Service is not ready", status_code=503, code="not_ready")
    return runtime


async def _noop_emit(event: str, data: dict[str, Any]) -> None:
    """Default emitter used when no progress listener is attached."""
    return None


def get_emit(config: RunnableConfig) -> EmitFn:
    """Returns the progress emitter from the run config, or a no-op.

    Args:
        config: The LangGraph run config.

    Returns:
        An async callable taking (event, data).
    """
    return (config.get("configurable") or {}).get("emit") or _noop_emit


def get_client_id(request: Request) -> str | None:
    """FastAPI dependency that reads the anonymous browser id.

    The id comes from the X-Client-Id header, or from the client_id query
    parameter for EventSource, which cannot send headers.

    Args:
        request: The incoming request.

    Returns:
        The validated client id, or None when the client sent none.

    Raises:
        AppError: With status 400 when the id has an invalid format.
    """
    raw = request.headers.get("x-client-id") or request.query_params.get("client_id")
    if raw is None:
        return None
    if not CLIENT_ID_PATTERN.match(raw):
        raise AppError("Invalid client id", status_code=400, code="invalid_client_id")
    return raw


def require_client_id(request: Request) -> str:
    """FastAPI dependency for endpoints that need a client id.

    Args:
        request: The incoming request.

    Returns:
        The validated client id.

    Raises:
        AppError: With status 400 when no valid client id was sent.
    """
    client_id = get_client_id(request)
    if client_id is None:
        raise AppError("Client id is required", status_code=400, code="client_id_required")
    return client_id