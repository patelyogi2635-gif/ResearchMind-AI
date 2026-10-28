import asyncio
import logging

import httpx
from fastapi import Request
from langchain_core.runnables import RunnableConfig

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.context import Runtime

logger = logging.getLogger(__name__)


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
    """Cancels running jobs and closes the shared HTTP client.

    Args:
        runtime: The runtime created at startup.
    """
    tasks = [
        job.task
        for job in runtime.jobs.values()
        if job.task is not None and not job.task.done()
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await runtime.http.aclose()
    logger.info("runtime.closed cancelled_jobs=%d", len(tasks))


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