from fastapi import APIRouter, Depends, Query

from app.controllers.job_controller import cancel_conversation_jobs
from app.controllers.memory_controller import (
    delete_conversation,
    get_conversation_page,
    list_conversations,
    lookup_id,
)
from app.core.runtime import get_app_runtime, require_client_id
from app.schemas.common import ErrorResponse
from app.schemas.context import Runtime
from app.schemas.memory import ConversationListResponse, ConversationResponse, LookupResponse

router = APIRouter()

ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.get("/conversations", response_model=ConversationListResponse, responses=ERROR_RESPONSES)
async def list_client_conversations(
    limit: int = Query(20, ge=1, le=50),
    before: float | None = Query(None, description="Cursor from next_before"),
    runtime: Runtime = Depends(get_app_runtime),
    client_id: str = Depends(require_client_id),
) -> ConversationListResponse:
    """Lists the caller's conversations, newest first."""
    return await list_conversations(runtime, client_id, limit, before)


@router.get(
    "/conversations/{conversation_id}", response_model=ConversationResponse, responses=ERROR_RESPONSES
)
async def get_client_conversation(
    conversation_id: str,
    limit: int = Query(50, ge=1, le=100),
    before: float | None = Query(None, description="Cursor from next_before"),
    runtime: Runtime = Depends(get_app_runtime),
    client_id: str = Depends(require_client_id),
) -> ConversationResponse:
    """Returns a conversation with its latest messages, oldest first."""
    return await get_conversation_page(runtime, client_id, conversation_id, limit, before)


@router.delete("/conversations/{conversation_id}", status_code=204, responses=ERROR_RESPONSES)
async def delete_client_conversation(
    conversation_id: str,
    runtime: Runtime = Depends(get_app_runtime),
    client_id: str = Depends(require_client_id),
) -> None:
    """Deletes a conversation with all its messages and jobs, stopping running jobs first."""
    await get_conversation_page(runtime, client_id, conversation_id, 1, None)
    await cancel_conversation_jobs(runtime, conversation_id)
    await delete_conversation(runtime, client_id, conversation_id)


@router.get("/lookup/{identifier}", response_model=LookupResponse, responses=ERROR_RESPONSES)
async def lookup_any_id(
    identifier: str,
    runtime: Runtime = Depends(get_app_runtime),
    client_id: str = Depends(require_client_id),
) -> LookupResponse:
    """Finds a conversation from a conversation, job, message or request id."""
    return await lookup_id(runtime, client_id, identifier)