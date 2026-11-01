from pydantic import BaseModel, Field

from app.schemas.jobs import JobError
from app.schemas.research import ResearchRequest, ResearchResponse


class ChatRequest(ResearchRequest):
    conversation_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{8,64}$",
        description="Continue an existing conversation. Omit to start a new one",
    )


class ConversationContext(BaseModel):
    turns: list[str] = Field(default_factory=list)
    brief_summary: str | None = None


class MessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    role: str
    kind: str
    text: str = ""
    intent: str | None = None
    job_id: str | None = None
    request_id: str | None = None
    status: str | None = None
    created_at: float
    result: ResearchResponse | None = None
    error: JobError | None = None


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int
    active_job_id: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]
    next_before: float | None = None


class ConversationResponse(BaseModel):
    conversation: ConversationSummary
    messages: list[MessageResponse]
    has_more: bool
    next_before: float | None = None


class LookupResponse(BaseModel):
    matched: str
    conversation_id: str | None = None
    job_id: str | None = None
    message_id: str | None = None
    request_id: str | None = None