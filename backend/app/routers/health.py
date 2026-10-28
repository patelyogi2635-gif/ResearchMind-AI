from fastapi import APIRouter, Depends

from app.controllers.health_controller import get_health, get_readiness
from app.core.runtime import get_app_runtime
from app.schemas.common import ErrorResponse, HealthResponse, ReadinessResponse
from app.schemas.context import Runtime

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe used by the deployment platform."""
    return get_health()


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ErrorResponse}},
)
async def readiness(runtime: Runtime = Depends(get_app_runtime)) -> ReadinessResponse:
    """Readiness probe with queue and job counts. Does not call external providers."""
    return get_readiness(runtime)