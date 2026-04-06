"""
tests/test_mcp.py
-----------------
Smoke test for the Phase 4 MCP client and tool adapter.
Connects only to the filesystem server (most reliable, no external deps).

Run with: PYTHONUTF8=1 python tests/test_mcp.py
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, ".")

from config import config
from display.console import print_rule, console
from rich.text import Text
from mcp_client.server_configs import get_server_configs, ServerConfig
from mcp_client.client import MCPClient
from mcp_client.tool_adapter import build_langchain_tools


def ok(msg: str) -> None:
    console.print(Text(f"  ✔  {msg}", style="bold green"))


def fail(msg: str) -> None:
    console.print(Text(f"  ✘  {msg}", style="bold red"))


def info(msg: str) -> None:
    console.print(Text(f"  ·  {msg}", style="dim white"))


async def run() -> None:
    print_rule("Phase 4 — MCP Client & Tool Adapter Smoke Test")
    workspace = config.workspace

    # ------------------------------------------------------------------
    # 1. ServerConfig builds correctly
    # ------------------------------------------------------------------
    print_rule("1. ServerConfig construction")
    all_configs = get_server_configs(workspace)
    info(f"Total server configs: {len(all_configs)}")
    for cfg in all_configs:
        info(f"  {cfg.name}: {cfg.command} {' '.join(cfg.args[:2])} ...")
    ok("get_server_configs returned configs without error")

    # ------------------------------------------------------------------
    # 2. Connect to filesystem server only (fastest, no API key needed)
    # ------------------------------------------------------------------
    print_rule("2. Connect to filesystem MCP server")
    fs_config = next(c for c in all_configs if c.name == "filesystem")

    async with MCPClient([fs_config]) as client:
        servers = client.get_connected_servers()
        if "filesystem" in servers:
            ok(f"Connected to: {servers}")
        else:
            fail(f"filesystem not in connected servers: {servers}")
            return

        # ------------------------------------------------------------------
        # 3. List tools
        # ------------------------------------------------------------------
        print_rule("3. List tools from filesystem server")
        mcp_tools = client.list_all_tools()
        tool_count = client.get_tool_count()
        ok(f"{tool_count} tool(s) loaded")
        for t in mcp_tools:
            info(f"  [{t.name}]  {(t.description or '')[:80]}")

        # ------------------------------------------------------------------
        # 4. Build LangChain tool adapters
        # ------------------------------------------------------------------
        print_rule("4. Build LangChain tool adapters")
        lc_tools = build_langchain_tools(mcp_tools, client)
        ok(f"build_langchain_tools → {len(lc_tools)} BaseTool(s)")
        for t in lc_tools:
            schema_fields = list(t.args_schema.model_fields.keys())
            info(f"  [{t.name}]  schema fields: {schema_fields}")

        # ------------------------------------------------------------------
        # 5. Call a real tool — list_directory on the workspace
        # ------------------------------------------------------------------
        print_rule("5. Call list_directory tool")
        list_tool = next((t for t in lc_tools if "list" in t.name.lower()), None)
        if list_tool is None:
            fail("No list_directory-style tool found")
        else:
            try:
                result = await list_tool.arun({"path": workspace})
                lines = result.strip().splitlines()
                ok(f"list_directory returned {len(lines)} line(s)")
                for line in lines[:5]:
                    info(f"  {line}")
                if len(lines) > 5:
                    info(f"  ... ({len(lines) - 5} more)")
            except Exception as e:
                fail(f"list_directory call failed: {e}")

        # ------------------------------------------------------------------
        # 6. Verify bind_tools works with the provider factory
        # ------------------------------------------------------------------
        print_rule("6. bind_tools integration with provider factory")
        try:
            from providers import get_model
            model = get_model("groq", api_key=config.groq_api_key)
            bound = model.bind_tools(lc_tools)
            ok(f"model.bind_tools({len(lc_tools)} tools) → {type(bound).__name__}")
        except Exception as e:
            fail(f"bind_tools failed: {e}")

    print_rule("All MCP checks complete")


if __name__ == "__main__":
    asyncio.run(run())
