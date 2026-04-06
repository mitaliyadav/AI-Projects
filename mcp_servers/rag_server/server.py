"""
mcp_servers/rag_server/server.py
---------------------------------
Custom MCP server that exposes LangChain documentation search as a tool.
Started as a subprocess by MCPClient via stdio transport.

Tools exposed
-------------
search_langchain_docs(query, k=5)
    Embeds the query with Ollama, queries ChromaDB, and returns the top-k
    most relevant chunks from the pre-indexed LangChain documentation.

get_rag_status()
    Returns whether the ChromaDB index is ready and how many chunks it holds.
    Useful for the agent to check before trying to search.

Prerequisites
-------------
Run `python main.py --setup-rag` once to build the ChromaDB index.
The server starts fine without the index but returns a clear error on query.

Running standalone (for testing)
---------------------------------
    python mcp_servers/rag_server/server.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure the project root is on the path so we can import our modules
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
import mcp.types as types

from mcp_servers.rag_server.retriever import retrieve, db_ready, get_collection_stats

# ------------------------------------------------------------------
# MCP Server instance
# ------------------------------------------------------------------
server = Server("langchain-rag-server")


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Advertise available tools to the MCP client."""
    return [
        types.Tool(
            name="search_langchain_docs",
            description=(
                "Search the LangChain documentation using semantic similarity. "
                "Use this when you need information about LangChain concepts, "
                "APIs, chains, agents, RAG patterns, LCEL, or how to use "
                "specific LangChain classes and functions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language question or search term about LangChain. "
                            "Example: 'How do I create a RAG chain with LCEL?'"
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_rag_status",
            description=(
                "Check whether the LangChain documentation index is ready. "
                "Returns the number of indexed chunks and the index location. "
                "Call this before search_langchain_docs if uncertain whether "
                "the index has been built."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# ------------------------------------------------------------------
# Tool execution
# ------------------------------------------------------------------
@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict | None,
) -> list[types.TextContent]:
    """Dispatch tool calls to the appropriate handler."""

    args = arguments or {}

    if name == "search_langchain_docs":
        return await _search_langchain_docs(args)

    elif name == "get_rag_status":
        return await _get_rag_status()

    else:
        return [
            types.TextContent(
                type="text",
                text=f"Unknown tool: '{name}'. Available tools: search_langchain_docs, get_rag_status",
            )
        ]


async def _search_langchain_docs(args: dict) -> list[types.TextContent]:
    """Execute a semantic search against the ChromaDB index."""

    query: str = args.get("query", "").strip()
    k: int = min(int(args.get("k", 5)), 10)  # cap at 10 to avoid huge responses

    if not query:
        return [types.TextContent(type="text", text="Error: 'query' must not be empty.")]

    if not db_ready():
        return [
            types.TextContent(
                type="text",
                text=(
                    "The LangChain documentation index has not been built yet.\n"
                    "Run:  python main.py --setup-rag\n"
                    "Then restart the agent session."
                ),
            )
        ]

    try:
        # Run the blocking ChromaDB query in a thread pool so we don't
        # block the asyncio event loop
        loop = asyncio.get_event_loop()
        chunks: list[str] = await loop.run_in_executor(None, retrieve, query, k)

        if not chunks:
            return [
                types.TextContent(
                    type="text",
                    text=f"No relevant documentation found for query: '{query}'",
                )
            ]

        # Format chunks with separators for readability
        formatted = f"Search results for: '{query}'\n\n"
        formatted += "\n\n─────────────────────────────────\n\n".join(
            f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
        )

        return [types.TextContent(type="text", text=formatted)]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"Error during search: {e}",
            )
        ]


async def _get_rag_status() -> list[types.TextContent]:
    """Return current index status."""
    loop = asyncio.get_event_loop()
    stats: dict = await loop.run_in_executor(None, get_collection_stats)

    if stats["status"] == "ready":
        text = (
            f"LangChain docs index is READY.\n"
            f"  Chunks indexed : {stats['count']}\n"
            f"  Collection     : {stats['collection']}\n"
            f"  Location       : {stats['path']}"
        )
    elif stats["status"] == "not_built":
        text = (
            "LangChain docs index is NOT built.\n"
            "Run:  python main.py --setup-rag"
        )
    else:
        text = f"Index error: {stats.get('error', 'unknown')}"

    return [types.TextContent(type="text", text=text)]


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="langchain-rag-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
