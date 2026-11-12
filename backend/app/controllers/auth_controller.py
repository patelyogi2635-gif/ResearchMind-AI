import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo.errors import DuplicateKeyError, PyMongoError

from app.controllers.memory_controller import new_id
from app.core.errors import AppError
from app.core.security import (
    DUMMY_HASH,
    create_access_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token,
    verify_password,
)
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.schemas.context import Runtime

logger = logging.getLogger(__name__)

MAX_TRACKED_LOGIN_KEYS = 5000
INVALID_CREDENTIALS = "Invalid email or password."
INVALID_REFRESH = "Your session has expired. Please sign in again."


def _database(runtime: Runtime):
    """Returns the database or raises a 503 error when accounts are unavailable."""
    if runtime.mongo_db is None:
        raise AppError(
            "Sign-in is temporarily unavailable. Please try again.",
            status_code=503,
            code="auth_unavailable",
        )
    return runtime.mongo_db


async def _run(runtime: Runtime, operation) -> Any:
    """Awaits a database operation with a timeout, mapping failures to a 503 error.

    A duplicate key error is re-raised unchanged so callers can handle it.
    """
    try:
        return await asyncio.wait_for(operation, timeout=runtime.settings.mongodb_timeout_ms / 1000)
    except DuplicateKeyError:
        raise
    except (PyMongoError, asyncio.TimeoutError) as exc:
        logger.error("auth.db_failed error=%s", str(exc)[:200])
        raise AppError(
            "Sign-in is temporarily unavailable. Please try again.",
            status_code=503,
            code="auth_unavailable",
        ) from exc


def _user_response(user: dict) -> UserResponse:
    """Maps a user document to its public representation."""
    return UserResponse(user_id=user["_id"], email=user["email"], name=user.get("name"))


async def _issue_tokens(runtime: Runtime, user: dict) -> TokenResponse:
    """Creates an access token and a new stored refresh token for a user.

    Only the SHA-256 digest of the refresh token is stored.

    Args:
        runtime: Shared runtime.
        user: The user document.

    Returns:
        The token pair with the public user profile.
    """
    settings = runtime.settings
    access_token, expires_in = create_access_token(
        user["_id"], runtime.jwt_secret, settings.access_token_minutes
    )
    refresh_token = new_refresh_token()
    now = time.time()
    session = {
        "_id": hash_refresh_token(refresh_token),
        "user_id": user["_id"],
        "revoked": False,
        "created_at": now,
        "expires_ts": now + settings.refresh_token_days * 86400,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_days),
    }
    await _run(runtime, _database(runtime).sessions.insert_one(session))
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        user=_user_response(user),
    )


def _recent_failures(runtime: Runtime, key: str) -> list[float]:
    """Returns the failed login timestamps for a key that are still inside the lock window."""
    window = runtime.settings.login_lock_minutes * 60
    cutoff = time.time() - window
    recent = [stamp for stamp in runtime.login_failures.get(key, []) if stamp > cutoff]
    if recent:
        runtime.login_failures[key] = recent
    else:
        runtime.login_failures.pop(key, None)
    return recent


def _check_login_allowed(runtime: Runtime, key: str) -> None:
    """Raises a 429 error when a key has too many recent failed logins.

    Raises:
        AppError: With status 429 and the remaining lock time in the message.
    """
    recent = _recent_failures(runtime, key)
    if len(recent) >= runtime.settings.login_max_attempts:
        wait = max(1, int((recent[0] + runtime.settings.login_lock_minutes * 60 - time.time()) / 60) + 1)
        raise AppError(
            f"Too many failed attempts. Try again in about {wait} minute(s).",
            status_code=429,
            code="too_many_attempts",
        )


def _record_login_failure(runtime: Runtime, key: str) -> None:
    """Records a failed login and keeps the tracking table bounded."""
    if len(runtime.login_failures) >= MAX_TRACKED_LOGIN_KEYS:
        for stale_key in list(runtime.login_failures):
            _recent_failures(runtime, stale_key)
    runtime.login_failures.setdefault(key, []).append(time.time())


async def register_user(runtime: Runtime, payload: RegisterRequest) -> TokenResponse:
    """Creates an account and signs the user in.

    Args:
        runtime: Shared runtime.
        payload: The validated registration data.

    Returns:
        A token pair for the new user.

    Raises:
        AppError: With status 403 when registration is disabled, 409 when the email
            is already registered, or 503 when the database is unavailable.
    """
    if not runtime.settings.registration_enabled:
        raise AppError("Registration is closed.", status_code=403, code="registration_disabled")
    db = _database(runtime)
    now = time.time()
    user = {
        "_id": new_id(),
        "email": payload.email,
        "name": (payload.name or "").strip() or None,
        "password_hash": await hash_password(payload.password),
        "created_at": now,
        "last_login_at": now,
    }
    try:
        await _run(runtime, db.users.insert_one(user))
    except DuplicateKeyError:
        raise AppError(
            "An account with this email already exists.", status_code=409, code="email_taken"
        ) from None
    logger.info("auth.registered user_id=%s", user["_id"])
    return await _issue_tokens(runtime, user)


async def login_user(runtime: Runtime, payload: LoginRequest, ip: str) -> TokenResponse:
    """Signs a user in with email and password.

    Unknown emails go through the same password check as known ones, so response
    time does not reveal which emails are registered. Repeated failures lock the
    email and address combination for a while.

    Args:
        runtime: Shared runtime.
        payload: The validated login data.
        ip: The caller's address, used for the failed-attempt limit.

    Returns:
        A token pair for the user.

    Raises:
        AppError: With status 401 for wrong credentials, 429 when locked out, or
            503 when the database is unavailable.
    """
    key = f"{ip}|{payload.email}"
    _check_login_allowed(runtime, key)
    db = _database(runtime)
    user = await _run(runtime, db.users.find_one({"email": payload.email}))
    valid = await verify_password(payload.password, user["password_hash"] if user else DUMMY_HASH)
    if user is None or not valid:
        _record_login_failure(runtime, key)
        logger.warning("auth.login_failed ip=%s", ip)
        raise AppError(INVALID_CREDENTIALS, status_code=401, code="invalid_credentials")
    runtime.login_failures.pop(key, None)
    await _run(runtime, db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login_at": time.time()}}))
    logger.info("auth.login user_id=%s", user["_id"])
    return await _issue_tokens(runtime, user)


async def refresh_tokens(runtime: Runtime, refresh_token: str) -> TokenResponse:
    """Exchanges a refresh token for a new token pair and revokes the old token.

    Every refresh token works once. Presenting an already-used token is treated as
    theft: all sessions of that user are revoked.

    Args:
        runtime: Shared runtime.
        refresh_token: The refresh token issued at login or the last refresh.

    Returns:
        A new token pair.

    Raises:
        AppError: With status 401 when the token is unknown, used, revoked or
            expired, or 503 when the database is unavailable.
    """
    db = _database(runtime)
    token_hash = hash_refresh_token(refresh_token)
    session = await _run(runtime, db.sessions.find_one({"_id": token_hash}))
    if session is None or session["expires_ts"] <= time.time():
        raise AppError(INVALID_REFRESH, status_code=401, code="invalid_refresh_token")
    if session["revoked"]:
        await _run(runtime, db.sessions.update_many({"user_id": session["user_id"]}, {"$set": {"revoked": True}}))
        logger.warning("auth.refresh_reuse user_id=%s action=revoked_all_sessions", session["user_id"])
        raise AppError(INVALID_REFRESH, status_code=401, code="invalid_refresh_token")
    claimed = await _run(
        runtime,
        db.sessions.find_one_and_update({"_id": token_hash, "revoked": False}, {"$set": {"revoked": True}}),
    )
    user = await _run(runtime, db.users.find_one({"_id": session["user_id"]}))
    if claimed is None or user is None:
        raise AppError(INVALID_REFRESH, status_code=401, code="invalid_refresh_token")
    return await _issue_tokens(runtime, user)


async def logout_user(runtime: Runtime, refresh_token: str) -> None:
    """Revokes a refresh token. Unknown tokens are ignored, so logging out is idempotent."""
    db = _database(runtime)
    await _run(
        runtime,
        db.sessions.update_one({"_id": hash_refresh_token(refresh_token)}, {"$set": {"revoked": True}}),
    )


async def get_profile(runtime: Runtime, user_id: str) -> UserResponse:
    """Returns the profile of the authenticated user.

    Raises:
        AppError: With status 401 when the account no longer exists.
    """
    user = await _run(runtime, _database(runtime).users.find_one({"_id": user_id}))
    if user is None:
        raise AppError("Authentication required.", status_code=401, code="not_authenticated")
    return _user_response(user)