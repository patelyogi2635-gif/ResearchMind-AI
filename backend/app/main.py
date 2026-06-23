import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import setup_logging
from app.routers import health, research

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger.info("Research Agent starting — env: %s", settings.app_env)
    logger.info("Startup complete. Graphs ready.")
    app.state.mcp_status = {"status": "not_checked_yet"}
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Research Agent API",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.app_env == "development" else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # allow all origins for development
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(research.router, prefix="/api", tags=["research"])

    return app


app = create_app()