import logging
import os
import sys

from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_npx_path() -> str:
    """
    On Windows, npx might not be on PATH inside a subprocess.
    Try to find it explicitly.
    """
    if sys.platform == "win32":
        # Common Windows locations for npx
        candidates = [
            "npx.cmd",   # Windows npm installs npx as npx.cmd
            "npx",
        ]
        return candidates[0]  # use npx.cmd on Windows
    return "npx"


def _build_mcp_config(settings) -> dict:
    npx = _get_npx_path()
    base_env = {**os.environ}

    return {
        "tavily": {
            "command": npx,
            "args": ["-y", "tavily-mcp"],    # no @latest
            "env": {**base_env, "TAVILY_API_KEY": settings.tavily_api_key},
            "transport": "stdio",
        },
        "github": {
            "command": npx,
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {**base_env, "GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token},
            "transport": "stdio",
        },
    }

async def get_mcp_tools(servers: list[str] | None = None) -> list:
    """
    Connect to MCP servers and return tools.

    Args:
        servers: list of server names to connect to.
                 None = connect to all servers.
                 ["tavily"] = only Tavily.
                 ["github"] = only GitHub.

    Returns: flat list of LangChain BaseTool objects.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    settings = get_settings()
    full_config = _build_mcp_config(settings)

    # Filter to requested servers only
    if servers:
        config = {k: v for k, v in full_config.items() if k in servers}
    else:
        config = full_config

    logger.info("Connecting to MCP servers: %s", list(config.keys()))

    client = MultiServerMCPClient(config)
    tools = await client.get_tools()

    logger.info("Tools loaded: %s", [t.name for t in tools])
    return tools


async def verify_mcp_connection() -> dict:
    """
    Tests each MCP server individually so we can pinpoint which one fails.
    Called by GET /api/health/mcp
    """
    result = {}

    # Test Tavily alone
    try:
        logger.info("Testing Tavily MCP server...")
        tools = await get_mcp_tools(servers=["tavily"])
        result["tavily"] = {
            "status": "connected",
            "tools": [t.name for t in tools],
        }
        logger.info("Tavily: OK — %s tools", len(tools))
    except Exception as e:
        logger.error("Tavily MCP failed: %s", e)
        result["tavily"] = {"status": "failed", "error": str(e)}

    # Test GitHub alone
    try:
        logger.info("Testing GitHub MCP server...")
        tools = await get_mcp_tools(servers=["github"])
        result["github"] = {
            "status": "connected",
            "tools": [t.name for t in tools],
        }
        logger.info("GitHub: OK — %s tools", len(tools))
    except Exception as e:
        logger.error("GitHub MCP failed: %s", e)
        result["github"] = {"status": "failed", "error": str(e)}

    result["total_connected"] = sum(
        1 for v in result.values()
        if isinstance(v, dict) and v.get("status") == "connected"
    )
    return result