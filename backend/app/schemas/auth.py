from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

EMAIL_PATTERN = r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$"
MIN_PASSWORD_CHARS = 8
MAX_PASSWORD_CHARS = 128


def _normalize_email(value: Any) -> Any:
    """Trims and lowercases an email so lookups are case-insensitive."""
    return value.strip().lower() if isinstance(value, str) else value


Email = Annotated[str, BeforeValidator(_normalize_email), Field(pattern=EMAIL_PATTERN, max_length=320)]


class RegisterRequest(BaseModel):
    email: Email
    password: str = Field(min_length=MIN_PASSWORD_CHARS, max_length=MAX_PASSWORD_CHARS)
    name: str | None = Field(default=None, max_length=60)


class LoginRequest(BaseModel):
    email: Email
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_CHARS)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


class UserResponse(BaseModel):
    user_id: str
    email: str
    name: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse