import logging

from app.core.errors import AppError
from app.core.mcp_client import get_mcp_tools
from app.schemas.common import (
    HealthResponse,
    McpHealthResponse,
    McpServerStatus,
    ToolInfo,
    ToolsResponse,
)

logger = logging.getLogger(__name__)

SERVICE_NAME = "research-agent"
MCP_SERVERS = ("tavily", "github")
TOOL_DESCRIPTION_CHARS = 120


def get_health() -> HealthResponse:
    """Returns the liveness status of the service."""
    return HealthResponse(status="ok", service=SERVICE_NAME)


async def check_mcp_servers() -> McpHealthResponse:
    """Checks every MCP server individually so a failure can be pinpointed.

    Returns:
        Per-server status plus the number of connected servers.
    """
    servers: dict[str, McpServerStatus] = {}
    for name in MCP_SERVERS:
        try:
            tools = await get_mcp_tools(servers=[name])
        except Exception as exc:
            logger.error("health.mcp_failed server=%s error=%s", name, exc)
            servers[name] = McpServerStatus(status="failed", error=type(exc).__name__)
            continue
        servers[name] = McpServerStatus(status="connected", tools=[tool.name for tool in tools])
    connected = sum(1 for status in servers.values() if status.status == "connected")
    return McpHealthResponse(servers=servers, total_connected=connected)


async def list_mcp_tools() -> ToolsResponse:
    """Lists the tools exposed by all MCP servers.

    Returns:
        Tool names with truncated descriptions.

    Raises:
        AppError: If the MCP servers cannot be reached.
    """
    try:
        tools = await get_mcp_tools()
    except Exception as exc:
        raise AppError("MCP servers unavailable", status_code=502, code="mcp_unavailable") from exc
    return ToolsResponse(
        tools=[
            ToolInfo(name=tool.name, description=(tool.description or "")[:TOOL_DESCRIPTION_CHARS])
            for tool in tools
        ]
    )
