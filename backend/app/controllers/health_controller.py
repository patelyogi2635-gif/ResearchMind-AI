from app.core.errors import AppError
from app.core.mongo import ping_mongo
from app.schemas.common import HealthResponse, ReadinessResponse
from app.schemas.context import Runtime
from app.schemas.jobs import JobStatus

SERVICE_NAME = "research-agent"


def get_health() -> HealthResponse:
    """Returns the liveness status of the service."""
    return HealthResponse(status="ok", service=SERVICE_NAME)


async def get_readiness(runtime: Runtime) -> ReadinessResponse:
    """Reports whether the service can accept work, with a job summary.

    Readiness inspects in-process state and pings MongoDB with a short timeout, so
    probes stay cheap and never call external providers.

    Args:
        runtime: The shared runtime.

    Returns:
        Queue and job counts.

    Raises:
        AppError: If the shared HTTP client has been closed.
    """
    if runtime.http.is_closed:
        raise AppError("Service is shutting down", status_code=503, code="not_ready")
    jobs = list(runtime.jobs.values())
    if not runtime.settings.mongodb_enabled:
        mongodb = "disabled"
    elif runtime.mongo_db is None:
        mongodb = "unavailable"
    else:
        mongodb = "connected" if await ping_mongo(runtime) else "down"
    return ReadinessResponse(
        status="ready",
        queued_jobs=sum(1 for job in jobs if job.status == JobStatus.QUEUED),
        running_jobs=sum(1 for job in jobs if job.status == JobStatus.RUNNING),
        stored_jobs=len(jobs),
        mongodb=mongodb,
        streaming=runtime.settings.streaming_enabled,
    )