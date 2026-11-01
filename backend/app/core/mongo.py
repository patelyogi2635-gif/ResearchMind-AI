import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError

from app.core.config import Settings
from app.schemas.context import Runtime

logger = logging.getLogger(__name__)

APP_NAME = "researchmind-ai"
PING_TIMEOUT_SECONDS = 2.0


def expiry_time(settings: Settings) -> datetime:
    """Returns the UTC time at which a newly written document should expire."""
    return datetime.now(timezone.utc) + timedelta(days=settings.mongodb_retention_days)


async def ensure_indexes(db) -> None:
    """Creates the indexes used for lookups by id, listing and retention.

    Args:
        db: The application database.
    """
    await db.conversations.create_index([("client_id", 1), ("updated_at", -1)], name="client_updated")
    await db.conversations.create_index("expires_at", expireAfterSeconds=0, name="ttl_expires")
    await db.messages.create_index([("conversation_id", 1), ("created_at", 1)], name="conversation_created")
    await db.messages.create_index([("job_id", 1), ("role", 1)], unique=True, name="job_role_unique")
    await db.messages.create_index("request_id", name="request_id")
    await db.messages.create_index("expires_at", expireAfterSeconds=0, name="ttl_expires")
    await db.jobs.create_index("conversation_id", name="conversation")
    await db.jobs.create_index("request_id", name="request_id")
    await db.jobs.create_index("expires_at", expireAfterSeconds=0, name="ttl_expires")


async def connect_mongo(runtime: Runtime) -> None:
    """Connects to MongoDB once at startup and attaches it to the runtime.

    The connection string is never logged. When MongoDB is unreachable the app
    starts without memory, unless MONGODB_REQUIRED is set.

    Args:
        runtime: The shared runtime to attach the database to.

    Raises:
        PyMongoError: If MongoDB is unreachable and MONGODB_REQUIRED is true.
    """
    settings = runtime.settings
    if not settings.mongodb_enabled:
        logger.info("startup.mongodb disabled reason=MONGODB_ENABLED=false mode=stateless")
        return

    started = time.perf_counter()
    client = AsyncMongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
        connectTimeoutMS=settings.mongodb_timeout_ms,
        appname=APP_NAME,
    )
    try:
        await client.admin.command("ping")
        ping_ms = (time.perf_counter() - started) * 1000
        db = client[settings.mongodb_db]
        await ensure_indexes(db)
    except PyMongoError as exc:
        await client.close()
        if settings.mongodb_required:
            logger.error("startup.mongodb failed required=true error=%s", str(exc)[:200])
            raise
        logger.warning("startup.mongodb unavailable error=%s mode=stateless", str(exc)[:200])
        return

    runtime.mongo_client = client
    runtime.mongo_db = db
    logger.info(
        "startup.mongodb connected db=%s ping_ms=%.0f indexes=ok retention_days=%d",
        settings.mongodb_db,
        ping_ms,
        settings.mongodb_retention_days,
    )


async def ping_mongo(runtime: Runtime) -> bool:
    """Returns whether MongoDB answers a ping within a short timeout."""
    if runtime.mongo_client is None:
        return False
    try:
        await asyncio.wait_for(runtime.mongo_client.admin.command("ping"), PING_TIMEOUT_SECONDS)
    except (PyMongoError, asyncio.TimeoutError):
        return False
    return True


async def close_mongo(runtime: Runtime) -> None:
    """Closes the MongoDB client if one was opened."""
    if runtime.mongo_client is None:
        return
    await runtime.mongo_client.close()
    runtime.mongo_client = None
    runtime.mongo_db = None
    logger.info("shutdown.mongodb closed")