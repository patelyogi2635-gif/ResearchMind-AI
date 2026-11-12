import asyncio
import os
import time

os.environ.setdefault("GROQ_API_KEY", "test-groq")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily")
os.environ.setdefault("GITHUB_TOKEN", "test-github")
os.environ.setdefault("MONGODB_ENABLED", "false")
os.environ.setdefault("JWT_SECRET", "t" * 40)

import jwt as pyjwt
import pytest
import pytest_asyncio
from pydantic import ValidationError

from app.controllers import auth_controller
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.mongo import connect_mongo
from app.core.runtime import close_runtime, create_runtime
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token,
    verify_password,
)
from app.schemas.auth import LoginRequest, RegisterRequest

TEST_DB = "researchmind_auth_test"
SECRET = "k" * 40
PASSWORD = "correct horse battery"


@pytest.mark.asyncio
async def test_password_hash_roundtrip_uses_a_fresh_salt_each_time():
    first = await hash_password(PASSWORD)
    second = await hash_password(PASSWORD)
    assert first != second
    assert first.startswith("$argon2")
    assert await verify_password(PASSWORD, first) is True
    assert await verify_password("wrong password", first) is False


@pytest.mark.asyncio
async def test_verify_password_rejects_an_unusable_hash():
    assert await verify_password(PASSWORD, "not-a-hash") is False


def test_access_token_roundtrip():
    token, expires_in = create_access_token("user-1", SECRET, 30)
    assert decode_access_token(token, SECRET) == "user-1"
    assert expires_in == 1800


def test_expired_access_token_is_rejected():
    token, _ = create_access_token("user-1", SECRET, -1)
    with pytest.raises(AppError) as exc_info:
        decode_access_token(token, SECRET)
    assert exc_info.value.status_code == 401
    assert exc_info.value.code == "token_expired"


def test_tampered_and_foreign_tokens_are_rejected():
    token, _ = create_access_token("user-1", SECRET, 30)
    for bad in (token, "not-a-token", token[:-3] + "abc"):
        with pytest.raises(AppError) as exc_info:
            decode_access_token(bad, "z" * 40)
        assert exc_info.value.code == "invalid_token"


def test_a_token_of_another_type_cannot_be_used_as_an_access_token():
    now = int(time.time())
    token = pyjwt.encode({"sub": "user-1", "iat": now, "exp": now + 60, "typ": "refresh"}, SECRET, algorithm="HS256")
    with pytest.raises(AppError) as exc_info:
        decode_access_token(token, SECRET)
    assert exc_info.value.code == "invalid_token"


def test_refresh_tokens_are_unique_and_hash_deterministically():
    first, second = new_refresh_token(), new_refresh_token()
    assert first != second
    assert len(first) >= 60
    assert hash_refresh_token(first) == hash_refresh_token(first)
    assert hash_refresh_token(first) != hash_refresh_token(second)


def test_register_request_normalizes_the_email_and_validates_the_fields():
    request = RegisterRequest(email="  Alice@Example.COM ", password="longenough")
    assert request.email == "alice@example.com"
    for bad_email in ("no-at-sign", "a@b", "two words@example.com", ""):
        with pytest.raises(ValidationError):
            RegisterRequest(email=bad_email, password="longenough")
    with pytest.raises(ValidationError):
        RegisterRequest(email="alice@example.com", password="short")
    with pytest.raises(ValidationError):
        LoginRequest(email="alice@example.com", password="")


def test_create_runtime_rejects_a_short_jwt_secret():
    with pytest.raises(ValueError):
        create_runtime(get_settings().model_copy(update={"jwt_secret": "too-short"}))


@pytest.mark.asyncio
async def test_login_lockout_applies_per_key_and_expires():
    runtime = create_runtime(get_settings().model_copy(update={"login_max_attempts": 2}))
    try:
        auth_controller._record_login_failure(runtime, "1.1.1.1|a@example.com")
        auth_controller._check_login_allowed(runtime, "1.1.1.1|a@example.com")
        auth_controller._record_login_failure(runtime, "1.1.1.1|a@example.com")
        with pytest.raises(AppError) as exc_info:
            auth_controller._check_login_allowed(runtime, "1.1.1.1|a@example.com")
        assert exc_info.value.status_code == 429
        assert exc_info.value.code == "too_many_attempts"
        auth_controller._check_login_allowed(runtime, "1.1.1.1|other@example.com")
        runtime.settings = runtime.settings.model_copy(update={"login_lock_minutes": 0})
        auth_controller._check_login_allowed(runtime, "1.1.1.1|a@example.com")
    finally:
        await close_runtime(runtime)


@pytest.mark.asyncio
async def test_login_failure_table_stays_bounded():
    runtime = create_runtime(get_settings().model_copy(update={"login_lock_minutes": 0}))
    try:
        for index in range(auth_controller.MAX_TRACKED_LOGIN_KEYS + 10):
            auth_controller._record_login_failure(runtime, f"ip|user{index}@example.com")
        assert len(runtime.login_failures) < auth_controller.MAX_TRACKED_LOGIN_KEYS
    finally:
        await close_runtime(runtime)


@pytest_asyncio.fixture
async def auth_runtime():
    settings = get_settings().model_copy(
        update={"mongodb_enabled": True, "mongodb_db": TEST_DB, "mongodb_timeout_ms": 1500, "login_max_attempts": 3}
    )
    runtime = create_runtime(settings)
    await connect_mongo(runtime)
    if runtime.mongo_db is None:
        await close_runtime(runtime)
        pytest.skip("MongoDB is not running on localhost")
    yield runtime
    await runtime.mongo_client.drop_database(TEST_DB)
    await close_runtime(runtime)


async def register(runtime, email="alice@example.com", password=PASSWORD, name="Alice"):
    return await auth_controller.register_user(runtime, RegisterRequest(email=email, password=password, name=name))


async def login(runtime, email="alice@example.com", password=PASSWORD, ip="127.0.0.1"):
    return await auth_controller.login_user(runtime, LoginRequest(email=email, password=password), ip)


@pytest.mark.asyncio
async def test_register_stores_a_hash_and_returns_a_working_token_pair(auth_runtime):
    tokens = await register(auth_runtime)
    assert decode_access_token(tokens.access_token, auth_runtime.jwt_secret) == tokens.user.user_id
    assert tokens.user.email == "alice@example.com"
    stored = await auth_runtime.mongo_db.users.find_one({"_id": tokens.user.user_id})
    assert stored["password_hash"].startswith("$argon2")
    assert PASSWORD not in str(stored)
    profile = await auth_controller.get_profile(auth_runtime, tokens.user.user_id)
    assert profile.name == "Alice"


@pytest.mark.asyncio
async def test_duplicate_email_is_rejected_case_insensitively(auth_runtime):
    await register(auth_runtime)
    with pytest.raises(AppError) as exc_info:
        await register(auth_runtime, email="ALICE@example.com")
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "email_taken"


@pytest.mark.asyncio
async def test_registration_can_be_closed(auth_runtime):
    auth_runtime.settings = auth_runtime.settings.model_copy(update={"registration_enabled": False})
    with pytest.raises(AppError) as exc_info:
        await register(auth_runtime)
    assert exc_info.value.status_code == 403
    assert exc_info.value.code == "registration_disabled"


@pytest.mark.asyncio
async def test_login_succeeds_and_wrong_credentials_give_one_generic_message(auth_runtime):
    registered = await register(auth_runtime)
    tokens = await login(auth_runtime)
    assert tokens.user.user_id == registered.user.user_id
    with pytest.raises(AppError) as wrong_password:
        await login(auth_runtime, password="not the password")
    with pytest.raises(AppError) as unknown_email:
        await login(auth_runtime, email="nobody@example.com")
    assert wrong_password.value.status_code == unknown_email.value.status_code == 401
    assert wrong_password.value.message == unknown_email.value.message


@pytest.mark.asyncio
async def test_repeated_failures_lock_the_account_even_for_the_right_password(auth_runtime):
    await register(auth_runtime)
    for _ in range(3):
        with pytest.raises(AppError):
            await login(auth_runtime, password="wrong password")
    with pytest.raises(AppError) as exc_info:
        await login(auth_runtime)
    assert exc_info.value.status_code == 429
    other_ip = await login(auth_runtime, ip="10.0.0.9")
    assert other_ip.access_token


@pytest.mark.asyncio
async def test_refresh_rotates_the_token_and_the_old_one_stops_working(auth_runtime):
    first = await register(auth_runtime)
    second = await auth_controller.refresh_tokens(auth_runtime, first.refresh_token)
    assert second.refresh_token != first.refresh_token
    assert second.user.user_id == first.user.user_id
    third = await auth_controller.refresh_tokens(auth_runtime, second.refresh_token)
    assert third.access_token


@pytest.mark.asyncio
async def test_reusing_a_rotated_token_revokes_every_session_of_that_user(auth_runtime):
    first = await register(auth_runtime)
    second = await auth_controller.refresh_tokens(auth_runtime, first.refresh_token)
    with pytest.raises(AppError) as reuse:
        await auth_controller.refresh_tokens(auth_runtime, first.refresh_token)
    assert reuse.value.status_code == 401
    with pytest.raises(AppError) as after_theft:
        await auth_controller.refresh_tokens(auth_runtime, second.refresh_token)
    assert after_theft.value.code == "invalid_refresh_token"


@pytest.mark.asyncio
async def test_refresh_rejects_unknown_and_expired_tokens(auth_runtime):
    with pytest.raises(AppError):
        await auth_controller.refresh_tokens(auth_runtime, new_refresh_token())
    tokens = await register(auth_runtime)
    await auth_runtime.mongo_db.sessions.update_one(
        {"_id": hash_refresh_token(tokens.refresh_token)}, {"$set": {"expires_ts": time.time() - 1}}
    )
    with pytest.raises(AppError) as exc_info:
        await auth_controller.refresh_tokens(auth_runtime, tokens.refresh_token)
    assert exc_info.value.code == "invalid_refresh_token"


@pytest.mark.asyncio
async def test_logout_revokes_the_refresh_token_and_is_idempotent(auth_runtime):
    tokens = await register(auth_runtime)
    await auth_controller.logout_user(auth_runtime, tokens.refresh_token)
    await auth_controller.logout_user(auth_runtime, tokens.refresh_token)
    await auth_controller.logout_user(auth_runtime, new_refresh_token())
    with pytest.raises(AppError):
        await auth_controller.refresh_tokens(auth_runtime, tokens.refresh_token)


@pytest.mark.asyncio
async def test_email_uniqueness_is_enforced_by_an_index(auth_runtime):
    indexes = await auth_runtime.mongo_db.users.index_information()
    sessions = await auth_runtime.mongo_db.sessions.index_information()
    assert indexes["email_unique"]["unique"] is True
    assert sessions["ttl_expires"]["expireAfterSeconds"] == 0


@pytest.mark.asyncio
async def test_two_users_cannot_see_each_others_conversations(auth_runtime):
    from app.controllers import memory_controller

    alice = await register(auth_runtime)
    bob = await register(auth_runtime, email="bob@example.com", name="Bob")
    conversation = await memory_controller.ensure_conversation(auth_runtime, alice.user.user_id, None, "alice only")
    assert (await memory_controller.list_conversations(auth_runtime, alice.user.user_id, 20, None)).conversations
    assert (await memory_controller.list_conversations(auth_runtime, bob.user.user_id, 20, None)).conversations == []
    with pytest.raises(AppError) as exc_info:
        await memory_controller.get_conversation_page(auth_runtime, bob.user.user_id, conversation["_id"], 10, None)
    assert exc_info.value.status_code == 404