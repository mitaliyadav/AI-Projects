"""
mcp/client.py
-------------
Async MCP client that connects to multiple MCP servers over stdio transport,
collects their tools, and routes tool calls to the correct server session.

Design
------
- MCPClient is an async context manager. Enter it once at startup (main.py),
  keep it alive for the whole session, and exit on shutdown.
- Each server is connected independently; a failing server emits a warning
  but does NOT abort the others.
- Tool names are globally unique across servers. If two servers expose a tool
  with the same name, the first one wins and a warning is logged.

Usage
-----
    async with MCPClient(configs) as client:
        tools = client.list_all_tools()      # list[mcp.types.Tool]
        result = await client.call_tool("read_file", {"path": "/tmp/x.py"})
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool

from mcp_client.server_configs import ServerConfig


# ------------------------------------------------------------------
# Internal bookkeeping per connected server
# ------------------------------------------------------------------
@dataclass
class _ServerEntry:
    name: str
    session: ClientSession
    tools: list[MCPTool]

    def has_tool(self, tool_name: str) -> bool:
        return any(t.name == tool_name for t in self.tools)


# ------------------------------------------------------------------
# MCPClient
# ------------------------------------------------------------------
class MCPClient:
    """
    Manages connections to all MCP servers for a single agent session.

    Parameters
    ----------
    server_configs : List of ServerConfig objects (from server_configs.py).
    """

    def __init__(self, server_configs: list[ServerConfig]) -> None:
        self._configs = server_configs
        self._exit_stack = AsyncExitStack()
        self._servers: list[_ServerEntry] = []
        self._tool_index: dict[str, _ServerEntry] = {}  # tool_name → server

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "MCPClient":
        await self._connect_all()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._exit_stack.aclose()
        self._servers.clear()
        self._tool_index.clear()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    async def _connect_all(self) -> None:
        """Connect to every configured server, skipping failures gracefully."""
        for cfg in self._configs:
            await self._connect_one(cfg)

    async def _connect_one(self, cfg: ServerConfig) -> None:
        """Attempt to connect to a single server. Logs on failure, does not raise."""
        try:
            stdio_params = cfg.to_stdio_params()

            read, write = await self._exit_stack.enter_async_context(
                stdio_client(stdio_params)
            )
            session: ClientSession = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            tools_result = await session.list_tools()
            tools: list[MCPTool] = tools_result.tools

            entry = _ServerEntry(name=cfg.name, session=session, tools=tools)
            self._servers.append(entry)

            # Build index — first server to register a name wins
            for tool in tools:
                if tool.name in self._tool_index:
                    existing = self._tool_index[tool.name].name
                    print(
                        f"[mcp] Warning: tool '{tool.name}' from '{cfg.name}' "
                        f"conflicts with '{existing}' — keeping '{existing}' version."
                    )
                else:
                    self._tool_index[tool.name] = entry

        except Exception as exc:
            # One server failing should not bring down the whole client
            print(
                f"[mcp] Warning: could not connect to server '{cfg.name}': {exc}"
            )

    # ------------------------------------------------------------------
    # Tool listing
    # ------------------------------------------------------------------
    def list_all_tools(self) -> list[MCPTool]:
        """
        Return all tools from all connected servers.
        Each tool is a native mcp.types.Tool with .name, .description,
        and .inputSchema.
        """
        tools: list[MCPTool] = []
        for server in self._servers:
            tools.extend(server.tools)
        return tools

    def get_connected_servers(self) -> list[str]:
        """Return the names of all successfully connected servers."""
        return [s.name for s in self._servers]

    def get_tool_count(self) -> int:
        """Return total number of tools available across all servers."""
        return len(self._tool_index)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """
        Call a tool by name and return its output as a string.

        Parameters
        ----------
        tool_name : Exact name of the MCP tool to invoke.
        args      : Keyword arguments matching the tool's inputSchema.

        Returns
        -------
        String result extracted from the tool's response content.

        Raises
        ------
        ValueError : If no connected server exposes a tool with that name.
        RuntimeError : If the tool call itself returns an error.
        """
        entry = self._tool_index.get(tool_name)
        if entry is None:
            available = list(self._tool_index.keys())
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {available}"
            )

        result = await entry.session.call_tool(tool_name, arguments=args)

        if result.isError:
            raise RuntimeError(
                f"Tool '{tool_name}' returned an error: "
                + _extract_text(result.content)
            )

        return _extract_text(result.content)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _extract_text(content: list[Any]) -> str:
    """
    Extract plain text from an MCP tool result's content list.
    Handles TextContent, and falls back to str() for other types.
    """
    parts: list[str] = []
    for item in content:
        if hasattr(item, "text") and isinstance(item.text, str):
            parts.append(item.text)
        elif hasattr(item, "data"):
            parts.append(str(item.data))
        else:
            parts.append(str(item))
    return "\n".join(parts) if parts else "(no content)"
