import asyncio
import hashlib
import secrets
import time
import uuid

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.core.errors import AppError

ALGORITHM = "HS256"
ACCESS_TOKEN_TYPE = "access"
MIN_JWT_SECRET_CHARS = 32

_hasher = PasswordHasher()
DUMMY_HASH = _hasher.hash("timing-equalizer-password")


async def hash_password(password: str) -> str:
    """Hashes a password with Argon2 in a worker thread.

    Hashing is deliberately slow and CPU-bound, so it must not run on the event
    loop where it would stall every other request.

    Args:
        password: The plain-text password.

    Returns:
        The encoded Argon2 hash, including its salt and parameters.
    """
    return await asyncio.to_thread(_hasher.hash, password)


async def verify_password(password: str, password_hash: str) -> bool:
    """Checks a password against a stored hash in a worker thread.

    Args:
        password: The plain-text password to check.
        password_hash: The stored Argon2 hash.

    Returns:
        True when the password matches, False for a mismatch or an unusable hash.
    """
    try:
        return await asyncio.to_thread(_hasher.verify, password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(user_id: str, secret: str, minutes: int) -> tuple[str, int]:
    """Creates a short-lived signed access token.

    Args:
        user_id: The user the token belongs to.
        secret: The signing secret.
        minutes: Token lifetime in minutes.

    Returns:
        The encoded token and its lifetime in seconds.
    """
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + minutes * 60,
        "typ": ACCESS_TOKEN_TYPE,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM), minutes * 60


def decode_access_token(token: str, secret: str) -> str:
    """Validates an access token and returns the user id it was issued to.

    Args:
        token: The encoded token.
        secret: The signing secret.

    Returns:
        The user id from the token's subject.

    Raises:
        AppError: With status 401 when the token is expired, tampered with or of
            the wrong type.
    """
    try:
        payload = jwt.decode(
            token, secret, algorithms=[ALGORITHM], options={"require": ["exp", "sub", "iat"]}
        )
    except jwt.ExpiredSignatureError:
        raise AppError("Your session has expired.", status_code=401, code="token_expired") from None
    except jwt.PyJWTError:
        raise AppError("Invalid credentials.", status_code=401, code="invalid_token") from None
    if payload.get("typ") != ACCESS_TOKEN_TYPE:
        raise AppError("Invalid credentials.", status_code=401, code="invalid_token")
    return str(payload["sub"])


def new_refresh_token() -> str:
    """Returns a new random refresh token with 384 bits of entropy."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """Returns the SHA-256 digest used to store and look up a refresh token.

    Only the digest is stored, so a database leak does not expose usable tokens.
    """
    return hashlib.sha256(token.encode()).hexdigest()