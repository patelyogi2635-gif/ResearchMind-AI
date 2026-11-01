import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logger import request_id_ctx, setup_logging
from app.controllers.memory_controller import recover_interrupted_jobs
from app.controllers.writer_controller import WRITER_MAX_TOKENS, WRITER_TEMPERATURE
from app.core.llm import get_chat_llm, get_llm, get_router_llm
from app.core.mongo import connect_mongo
from app.core.runtime import close_runtime, create_runtime
from app.routers import conversations, health, research

logger = logging.getLogger(__name__)

APP_TITLE = "ResearchMind AI - R&D Assistant"
APP_VERSION = "0.5.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Creates every shared service once at startup and releases them at shutdown.

    Each service logs when it is ready and when it is closed. Shutdown runs in
    reverse order: running jobs first, then MongoDB, then the HTTP client.
    """
    started = time.perf_counter()
    settings = get_settings()
    logger.info("startup.begin env=%s version=%s", settings.app_env, APP_VERSION)

    runtime = create_runtime(settings)
    logger.info(
        "startup.http_client ready pool=%d keepalive=%d timeout=%.0fs",
        settings.http_max_connections,
        settings.http_max_keepalive_connections,
        settings.http_timeout_seconds,
    )

    get_llm(temperature=0.3)
    get_llm(temperature=WRITER_TEMPERATURE, max_tokens=WRITER_MAX_TOKENS)
    get_router_llm()
    get_chat_llm()
    logger.info(
        "startup.llm_clients ready writer=%s router=%s chat=%s streaming=%s",
        settings.llm_model,
        settings.router_model,
        settings.chat_model,
        settings.streaming_enabled,
    )
    logger.info(
        "startup.limits pipelines=%d searches=%d llm_calls=%d pending_jobs=%d",
        settings.max_concurrent_pipelines,
        settings.max_concurrent_searches,
        settings.max_concurrent_llm_calls,
        settings.max_pending_jobs,
    )

    await connect_mongo(runtime)
    await recover_interrupted_jobs(runtime)

    app.state.runtime = runtime
    logger.info(
        "startup.ready took=%.2fs mongodb=%s",
        time.perf_counter() - started,
        "connected" if runtime.mongo_db is not None else "stateless",
    )
    try:
        yield
    finally:
        logger.info("shutdown.begin running_jobs=%d", sum(1 for job in runtime.jobs.values() if job.task and not job.task.done()))
        await close_runtime(runtime)
        logger.info("shutdown.complete")


async def request_context_middleware(request: Request, call_next):
    """Assigns a request id, logs the request outcome and echoes the id in a header."""
    request_id = uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    token = request_id_ctx.set(request_id)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    finally:
        logger.info(
            "http.request method=%s path=%s status=%s duration_ms=%.0f",
            request.method,
            request.url.path,
            status_code,
            (time.perf_counter() - started) * 1000,
        )
        request_id_ctx.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


def create_app() -> FastAPI:
    """Builds the FastAPI application with middleware, handlers and routers."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.app_env == "development" else None,
    )

    app.middleware("http")(request_context_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(research.router, prefix="/api", tags=["research"])
    app.include_router(conversations.router, prefix="/api", tags=["conversations"])
    return app


app = create_app()