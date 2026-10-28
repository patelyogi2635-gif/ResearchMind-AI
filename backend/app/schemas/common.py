from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    details: Any = None


class ErrorResponse(BaseModel):
    detail: str
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadinessResponse(BaseModel):
    status: str
    queued_jobs: int
    running_jobs: int
    stored_jobs: int