import logging
from fastapi import APIRouter, Request
from app.mcp_client import verify_mcp_connection, get_mcp_tools

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "research-agent"}


@router.get("/health/mcp")
async def mcp_health():
    result = await verify_mcp_connection()
    return result


@router.get("/health/tools")
async def list_tools():
    tools = await get_mcp_tools()
    return {
        "tools": [
            {"name": t.name, "description": t.description[:120]}
            for t in tools
        ]
    }