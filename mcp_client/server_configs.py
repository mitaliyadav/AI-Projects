"""
mcp/server_configs.py
---------------------
Defines the stdio launch configuration for every MCP server the agent
connects to. Each server is started as a subprocess via the stdio transport.

Servers
-------
filesystem  — @modelcontextprotocol/server-filesystem (Node.js via npx)
              Gives the agent read/write/search access to a local workspace.

context7    — @upstash/context7-mcp (Node.js via npx)
              Resolves library names to up-to-date documentation snippets.

rag         — mcp_servers/rag_server/server.py (local Python process)
              Custom semantic-search server over LangChain documentation.
              Must be indexed first with: python main.py --setup-rag
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mcp import StdioServerParameters


@dataclass
class ServerConfig:
    """Launch configuration for a single MCP server subprocess."""

    name: str                          # Human-readable label shown in the UI
    command: str                       # Executable to run
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None  # Extra env vars merged with current env

    def to_stdio_params(self) -> StdioServerParameters:
        """Convert to the MCP SDK's StdioServerParameters."""
        merged_env: dict[str, str] | None = None
        if self.env:
            merged_env = {**os.environ, **self.env}
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=merged_env,
        )


def get_server_configs(workspace: str, project_root: str | None = None) -> list[ServerConfig]:
    """
    Build the list of MCP server configs for the current session.

    Parameters
    ----------
    workspace    : Absolute path the filesystem server is allowed to access.
    project_root : Root of this project (used to locate the RAG server script).
                   Defaults to the directory containing this file.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent

    # On Windows, npx is a .cmd file so we need cmd /c npx
    npx_cmd = "npx"
    if sys.platform == "win32":
        npx_cmd = "npx.cmd"

    return [
        # ------------------------------------------------------------------
        # 1. Filesystem server
        #    Provides: read_file, write_file, create_directory, list_directory,
        #              move_file, search_files, get_file_info, etc.
        # ------------------------------------------------------------------
        ServerConfig(
            name="filesystem",
            command=npx_cmd,
            args=["-y", "@modelcontextprotocol/server-filesystem", workspace],
        ),

        # ------------------------------------------------------------------
        # 2. Context7 — external documentation resolver
        #    Provides: resolve-library-id, get-library-docs
        # ------------------------------------------------------------------
        ServerConfig(
            name="context7",
            command=npx_cmd,
            args=["-y", "@upstash/context7-mcp"],
        ),

        # ------------------------------------------------------------------
        # 3. Custom RAG server — LangChain documentation search
        #    Provides: search_langchain_docs
        #    NOTE: Run `python main.py --setup-rag` once before first use.
        # ------------------------------------------------------------------
        ServerConfig(
            name="rag",
            command=sys.executable,          # same Python interpreter as main app
            args=[str(root / "mcp_servers" / "rag_server" / "server.py")],
        ),
    ]
