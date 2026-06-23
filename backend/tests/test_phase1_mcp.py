"""
tests/test_phase1_mcp.py — smoke tests for MCP server connections.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN:
  cd backend/
  pytest tests/test_phase1_mcp.py -v -s

The -s flag shows print() output so you can see the tool results.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These tests make REAL external API calls (Tavily + GitHub).
They require valid API keys in your .env file.
They are integration tests, not unit tests — intentionally.

Phase 1's goal IS to prove external connectivity works.
"""

import asyncio
import pytest
from app.mcp_client import get_mcp_tools, verify_mcp_connection


# ── Test 1: MCP servers start and tools load ──────────────────────────

@pytest.mark.asyncio
async def test_mcp_servers_connect():
    """
    Verifies that both MCP servers start successfully
    and expose at least one tool each.
    """
    async with get_mcp_tools() as tools:
        tool_names = [t.name for t in tools]

        print(f"\n✓ Loaded {len(tools)} MCP tools:")
        for name in tool_names:
            print(f"  - {name}")

        # We expect at least Tavily + a few GitHub tools
        assert len(tools) >= 2, f"Expected at least 2 tools, got {len(tools)}"


# ── Test 2: Tavily search returns results ─────────────────────────────

@pytest.mark.asyncio
async def test_tavily_search():
    """
    Calls the Tavily MCP tool with a real query.
    Verifies the response has the expected structure.
    """
    async with get_mcp_tools() as tools:
        tavily_tool = next(
            (t for t in tools if "tavily" in t.name.lower()),
            None
        )

        assert tavily_tool is not None, \
            "Tavily tool not found — check TAVILY_API_KEY in .env"

        print(f"\n✓ Found Tavily tool: {tavily_tool.name}")
        print(f"  Description: {tavily_tool.description[:100]}...")

        # Make the actual search call
        result = await tavily_tool.ainvoke({"query": "LangGraph multi-agent tutorial"})

        print(f"\n✓ Tavily search result (first 300 chars):")
        print(f"  {str(result)[:300]}")

        assert result is not None
        assert len(str(result)) > 50, "Result seems too short — something may be wrong"


# ── Test 3: GitHub search returns repos ───────────────────────────────

@pytest.mark.asyncio
async def test_github_search():
    """
    Calls the GitHub MCP tool with a real query.
    Verifies repositories are returned.
    """
    async with get_mcp_tools() as tools:
        # GitHub MCP exposes several tools — find the repo search one
        github_tool = next(
            (t for t in tools if "search_repositories" in t.name.lower()),
            None
        )

        assert github_tool is not None, \
            "GitHub search_repositories tool not found — check GITHUB_TOKEN in .env"

        print(f"\n✓ Found GitHub tool: {github_tool.name}")

        result = await github_tool.ainvoke({"query": "langgraph python"})

        print(f"\n✓ GitHub search result (first 300 chars):")
        print(f"  {str(result)[:300]}")

        assert result is not None


# ── Test 4: verify_mcp_connection helper ──────────────────────────────

@pytest.mark.asyncio
async def test_verify_mcp_connection():
    """Tests the health-check helper used by the /health/mcp endpoint."""
    status = await verify_mcp_connection()

    print(f"\n✓ MCP connection status:")
    for key, val in status.items():
        print(f"  {key}: {val}")

    assert "error" not in status, f"MCP connection error: {status.get('error')}"
    assert status.get("total_tools", 0) >= 2