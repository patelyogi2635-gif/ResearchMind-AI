import logging
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _npx_command() -> str:
    """Returns the npx executable name for the current platform."""
    return "npx.cmd" if sys.platform == "win32" else "npx"


def _build_mcp_config(settings: Settings) -> dict[str, dict]:
    """Builds the stdio MCP server configuration for every supported server."""
    npx = _npx_command()
    base_env = {**os.environ}
    return {
        "tavily": {
            "command": npx,
            "args": ["-y", "tavily-mcp"],
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
    """Connects to MCP servers and returns their tools as LangChain tools.

    Args:
        servers: Server names to connect to. None connects to all servers.

    Returns:
        A flat list of LangChain BaseTool objects.

    Raises:
        ValueError: If none of the requested servers are known.
    """
    full_config = _build_mcp_config(get_settings())
    config = {k: v for k, v in full_config.items() if k in servers} if servers else full_config
    if not config:
        raise ValueError(f"Unknown MCP servers requested: {servers}")

    logger.info("mcp.connect servers=%s", list(config))
    client = MultiServerMCPClient(config)
    tools = await client.get_tools()
    logger.info("mcp.tools_loaded names=%s", [tool.name for tool in tools])
    return tools
