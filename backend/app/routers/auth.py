from fastapi import APIRouter, Depends, Request

from app.controllers.auth_controller import (
    get_profile,
    login_user,
    logout_user,
    refresh_tokens,
    register_user,
)
from app.core.runtime import get_app_runtime, get_current_user_id
from app.schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse, UserResponse
from app.schemas.common import ErrorResponse
from app.schemas.context import Runtime

router = APIRouter()

ERROR_RESPONSES = {
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def _client_ip(request: Request) -> str:
    """Returns the caller's address, or 'unknown' when it is not available."""
    return request.client.host if request.client else "unknown"


@router.post("/auth/register", response_model=TokenResponse, status_code=201, responses=ERROR_RESPONSES)
async def register(payload: RegisterRequest, runtime: Runtime = Depends(get_app_runtime)) -> TokenResponse:
    """Creates an account and returns a token pair."""
    return await register_user(runtime, payload)


@router.post("/auth/login", response_model=TokenResponse, responses=ERROR_RESPONSES)
async def login(
    payload: LoginRequest, request: Request, runtime: Runtime = Depends(get_app_runtime)
) -> TokenResponse:
    """Signs in with email and password and returns a token pair."""
    return await login_user(runtime, payload, _client_ip(request))


@router.post("/auth/refresh", response_model=TokenResponse, responses=ERROR_RESPONSES)
async def refresh(payload: RefreshRequest, runtime: Runtime = Depends(get_app_runtime)) -> TokenResponse:
    """Rotates a refresh token and returns a new token pair."""
    return await refresh_tokens(runtime, payload.refresh_token)


@router.post("/auth/logout", status_code=204, responses=ERROR_RESPONSES)
async def logout(payload: RefreshRequest, runtime: Runtime = Depends(get_app_runtime)) -> None:
    """Revokes the given refresh token."""
    await logout_user(runtime, payload.refresh_token)


@router.get("/auth/me", response_model=UserResponse, responses=ERROR_RESPONSES)
async def me(
    runtime: Runtime = Depends(get_app_runtime), user_id: str = Depends(get_current_user_id)
) -> UserResponse:
    """Returns the signed-in user's profile."""
    return await get_profile(runtime, user_id)