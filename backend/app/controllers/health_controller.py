from app.core.errors import AppError
from app.schemas.common import HealthResponse, ReadinessResponse
from app.schemas.context import Runtime
from app.schemas.jobs import JobStatus

SERVICE_NAME = "research-agent"


def get_health() -> HealthResponse:
    """Returns the liveness status of the service."""
    return HealthResponse(status="ok", service=SERVICE_NAME)


def get_readiness(runtime: Runtime) -> ReadinessResponse:
    """Reports whether the service can accept work, with a job summary.

    Readiness only inspects in-process state, so probes stay cheap and never call
    external providers.

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
    return ReadinessResponse(
        status="ready",
        queued_jobs=sum(1 for job in jobs if job.status == JobStatus.QUEUED),
        running_jobs=sum(1 for job in jobs if job.status == JobStatus.RUNNING),
        stored_jobs=len(jobs),
    )