from typing import Any

from pydantic import BaseModel, Field


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


class McpServerStatus(BaseModel):
    status: str
    tools: list[str] = Field(default_factory=list)
    error: str | None = None


class McpHealthResponse(BaseModel):
    servers: dict[str, McpServerStatus]
    total_connected: int


class ToolInfo(BaseModel):
    name: str
    description: str


class ToolsResponse(BaseModel):
    tools: list[ToolInfo]
