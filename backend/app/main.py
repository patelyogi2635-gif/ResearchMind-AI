import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logger import request_id_ctx, setup_logging
from app.routers import health, research

logger = logging.getLogger(__name__)

APP_TITLE = "ResearchMind AI - R&D Assistant"
APP_VERSION = "0.3.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Logs application startup and shutdown."""
    settings = get_settings()
    logger.info(
        "app.startup env=%s version=%s model=%s",
        settings.app_env,
        APP_VERSION,
        settings.llm_model,
    )
    yield
    logger.info("app.shutdown")


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
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(research.router, prefix="/api", tags=["research"])
    return app


app = create_app()
