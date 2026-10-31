import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chat import ChatReply
from app.schemas.research import ResearchResponse


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})
ACTIVE_STATUSES = frozenset({JobStatus.QUEUED, JobStatus.RUNNING})


class JobEvent(BaseModel):
    seq: int
    event: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float


class JobError(BaseModel):
    code: str
    message: str
    status_code: int


class JobRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    query: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    events: list[JobEvent] = Field(default_factory=list)
    result: ResearchResponse | None = None
    reply: ChatReply | None = None
    error: JobError | None = None
    task: asyncio.Task | None = None
    condition: asyncio.Condition


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    stream_url: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    query: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    events_count: int
    result: ResearchResponse | None = None
    reply: ChatReply | None = None
    error: JobError | None = None