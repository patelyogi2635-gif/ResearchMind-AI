from fastapi import APIRouter

from app.controllers.health_controller import check_mcp_servers, get_health, list_mcp_tools
from app.schemas.common import ErrorResponse, HealthResponse, McpHealthResponse, ToolsResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe used by the deployment platform."""
    return get_health()


@router.get("/health/mcp", response_model=McpHealthResponse)
async def mcp_health() -> McpHealthResponse:
    """Reports the connection status of each MCP server."""
    return await check_mcp_servers()


@router.get(
    "/health/tools",
    response_model=ToolsResponse,
    responses={502: {"model": ErrorResponse}},
)
async def mcp_tools() -> ToolsResponse:
    """Lists the tools exposed by the MCP servers."""
    return await list_mcp_tools()
