import asyncio

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.schemas.jobs import JobRecord


class Runtime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    http: httpx.AsyncClient
    search_semaphore: asyncio.Semaphore
    llm_semaphore: asyncio.Semaphore
    pipeline_semaphore: asyncio.Semaphore
    jobs: dict[str, JobRecord] = Field(default_factory=dict)