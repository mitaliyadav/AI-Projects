"""
tests/test_rag.py
-----------------
Smoke test for the Phase 5 Custom RAG MCP Server.
Tests the retriever directly and the MCP server as a subprocess.

Run with: PYTHONUTF8=1 python tests/test_rag.py
"""

import asyncio
import sys

sys.path.insert(0, ".")

from display.console import print_rule, console
from rich.text import Text
from mcp_servers.rag_server.retriever import db_ready, retrieve, retrieve_with_scores, get_collection_stats
from mcp_client.server_configs import get_server_configs
from mcp_client.client import MCPClient
from mcp_client.tool_adapter import build_langchain_tools
from config import config


def ok(msg: str) -> None:
    console.print(Text(f"  ✔  {msg}", style="bold green"))


def fail(msg: str) -> None:
    console.print(Text(f"  ✘  {msg}", style="bold red"))


def info(msg: str) -> None:
    console.print(Text(f"  ·  {msg}", style="dim white"))


async def run() -> None:
    print_rule("Phase 5 — Custom RAG MCP Server Smoke Test")

    # ------------------------------------------------------------------
    # 1. Verify ChromaDB exists
    # ------------------------------------------------------------------
    print_rule("1. ChromaDB index status")
    if not db_ready():
        fail("ChromaDB not found. Run: python main.py --setup-rag")
        return

    stats = get_collection_stats()
    ok(f"Index is READY — {stats['count']} chunks in '{stats['collection']}'")
    info(f"  Location: {stats['path']}")

    # ------------------------------------------------------------------
    # 2. Direct retriever — basic query
    # ------------------------------------------------------------------
    print_rule("2. Direct retriever query")
    try:
        query = "What is LCEL and how does it work?"
        chunks = retrieve(query, k=3)
        ok(f"retrieve() returned {len(chunks)} chunk(s) for: '{query}'")
        for i, chunk in enumerate(chunks, 1):
            preview = chunk[:120].replace("\n", " ")
            info(f"  [{i}] {preview}...")
    except Exception as e:
        fail(f"retrieve() failed: {e}")

    # ------------------------------------------------------------------
    # 3. Retriever with similarity scores
    # ------------------------------------------------------------------
    print_rule("3. Retriever with scores")
    try:
        query = "How do I build an agent with LangChain tools?"
        results = retrieve_with_scores(query, k=3)
        ok(f"retrieve_with_scores() returned {len(results)} result(s)")
        for i, (chunk, score) in enumerate(results, 1):
            preview = chunk[:80].replace("\n", " ")
            info(f"  [{i}] score={score:.4f}  {preview}...")
    except Exception as e:
        fail(f"retrieve_with_scores() failed: {e}")

    # ------------------------------------------------------------------
    # 4. Connect to RAG MCP server via MCPClient
    # ------------------------------------------------------------------
    print_rule("4. RAG MCP server via MCPClient")
    all_configs = get_server_configs(config.workspace)
    rag_config = next(c for c in all_configs if c.name == "rag")

    try:
        async with MCPClient([rag_config]) as client:
            servers = client.get_connected_servers()
            if "rag" not in servers:
                fail(f"RAG server not connected. Got: {servers}")
                return
            ok(f"Connected to RAG server: {servers}")

            # ------------------------------------------------------------------
            # 5. List RAG tools
            # ------------------------------------------------------------------
            print_rule("5. List tools from RAG server")
            mcp_tools = client.list_all_tools()
            ok(f"{len(mcp_tools)} tool(s) exposed by RAG server")
            for t in mcp_tools:
                info(f"  [{t.name}]  {(t.description or '')[:80]}")

            # ------------------------------------------------------------------
            # 6. Call search_langchain_docs via MCP
            # ------------------------------------------------------------------
            print_rule("6. Call search_langchain_docs via MCP")
            lc_tools = build_langchain_tools(mcp_tools, client)
            search_tool = next((t for t in lc_tools if t.name == "search_langchain_docs"), None)

            if search_tool is None:
                fail("search_langchain_docs tool not found in RAG server")
            else:
                result = await search_tool.arun({"query": "How does RAG work in LangChain?", "k": 2})
                ok(f"search_langchain_docs returned {len(result)} chars")
                info(f"  Preview: {result[:200].replace(chr(10), ' ')}...")

            # ------------------------------------------------------------------
            # 7. Call get_rag_status via MCP
            # ------------------------------------------------------------------
            print_rule("7. Call get_rag_status via MCP")
            status_tool = next((t for t in lc_tools if t.name == "get_rag_status"), None)
            if status_tool is None:
                fail("get_rag_status tool not found")
            else:
                status_result = await status_tool.arun({})
                ok("get_rag_status call succeeded")
                info(f"  {status_result.strip()}")

    except Exception as e:
        fail(f"RAG MCP server test failed: {e}")
        import traceback
        traceback.print_exc()

    print_rule("All RAG checks complete")


if __name__ == "__main__":
    asyncio.run(run())
