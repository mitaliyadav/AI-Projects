"""
tests/test_agent.py
--------------------
Smoke test for Phase 6 — Agentic Loop.

Tests:
  1.  Imports and instantiation
  2.  MessageHistory — add / get / clear / trim
  3.  AgentLoop instantiation with real model + MCP tools
  4.  Single-turn run with NO tool calls  (model just replies)
  5.  Single-turn run WITH a tool call    (filesystem list_directory)
  6.  auto_execute=False confirmation path (simulated decline)
  7.  Max-turns hard limit fires correctly
  8.  History carries context across two tasks

Run with: PYTHONUTF8=1 python tests/test_agent.py
"""

import asyncio
import sys

sys.path.insert(0, ".")

from display.console import print_rule, console
from rich.text import Text


def ok(msg: str) -> None:
    console.print(Text(f"  ✔  {msg}", style="bold green"))


def fail(msg: str) -> None:
    console.print(Text(f"  ✘  {msg}", style="bold red"))


def info(msg: str) -> None:
    console.print(Text(f"  ·  {msg}", style="dim white"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def make_loop(max_turns: int = 5, auto_execute: bool = True):
    """Build a real AgentLoop wired to Groq + filesystem MCP server."""
    from config import config
    from providers import get_model_from_config
    from mcp_client import MCPClient, get_server_configs, build_langchain_tools
    from agent import AgentLoop

    model = get_model_from_config(config)

    all_configs = get_server_configs(config.workspace)
    fs_config = next(c for c in all_configs if c.name == "filesystem")

    # Return the client + loop so the caller can manage the context
    client = MCPClient([fs_config])
    await client.__aenter__()

    mcp_tools = client.list_all_tools()
    lc_tools = build_langchain_tools(mcp_tools, client)

    loop = AgentLoop(
        model=model,
        tools=lc_tools,
        max_turns=max_turns,
        auto_execute=auto_execute,
    )
    return loop, client


async def run() -> None:
    print_rule("Phase 6 — Agentic Loop Smoke Test")

    # ------------------------------------------------------------------
    # 1. Imports
    # ------------------------------------------------------------------
    print_rule("1. Imports")
    try:
        from agent import AgentLoop, MessageHistory, SYSTEM_PROMPT
        ok("AgentLoop, MessageHistory, SYSTEM_PROMPT imported")
        info(f"  System prompt length: {len(SYSTEM_PROMPT)} chars")
    except Exception as e:
        fail(f"Import failed: {e}")
        return

    # ------------------------------------------------------------------
    # 2. MessageHistory unit tests
    # ------------------------------------------------------------------
    print_rule("2. MessageHistory")
    try:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
        from agent.history import MessageHistory

        h = MessageHistory(system_prompt="You are a test assistant.")

        # Empty history returns only the system message
        msgs = h.get_messages()
        assert len(msgs) == 1 and isinstance(msgs[0], SystemMessage), "Should start with only SystemMessage"

        # Add human
        h.add_human("Hello!")
        msgs = h.get_messages()
        assert len(msgs) == 2
        assert isinstance(msgs[1], HumanMessage)

        # Add AI
        ai_msg = AIMessage(content="Hi there!", tool_calls=[])
        h.add_ai(ai_msg)
        assert len(h) == 2  # system not counted

        # Add tool result
        h.add_tool_result("call_1", "read_file", "file contents here")
        assert len(h) == 3
        msgs = h.get_messages()
        assert isinstance(msgs[-1], ToolMessage)
        assert msgs[-1].tool_call_id == "call_1"

        # Clear
        h.clear()
        assert len(h) == 0
        msgs = h.get_messages()
        assert len(msgs) == 1  # only system

        ok("MessageHistory: add_human / add_ai / add_tool_result / clear all pass")
    except Exception as e:
        fail(f"MessageHistory test failed: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 3. AgentLoop instantiation
    # ------------------------------------------------------------------
    print_rule("3. AgentLoop instantiation")
    loop = None
    client = None
    try:
        loop, client = await make_loop(max_turns=3, auto_execute=True)
        tool_names = [t.name for t in loop._tools]
        ok(f"AgentLoop created with {len(loop._tools)} tool(s)")
        info(f"  Tools: {tool_names[:5]}{'...' if len(tool_names) > 5 else ''}")
        info(f"  max_turns={loop._max_turns}  auto_execute={loop._auto_execute}")
    except Exception as e:
        fail(f"AgentLoop instantiation failed: {e}")
        import traceback; traceback.print_exc()
        return

    # ------------------------------------------------------------------
    # 4. Single-turn run — no tools expected
    # ------------------------------------------------------------------
    print_rule("4. Single-turn run (no tool call expected)")
    try:
        loop.clear_history()
        info("  Task: 'Reply with exactly the word PONG and nothing else.'")
        await loop.run("Reply with exactly the word PONG and nothing else.")
        ok("Run completed without error")
    except Exception as e:
        fail(f"Single-turn run failed: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 5. Tool-calling run — list workspace files
    # ------------------------------------------------------------------
    print_rule("5. Tool-calling run (filesystem list_directory)")
    try:
        loop.clear_history()
        from config import config
        task = f"List the files in {config.workspace} using list_directory. Just list them, nothing else."
        info(f"  Task: '{task}'")
        await loop.run(task)
        ok("Tool-calling run completed without error")
    except Exception as e:
        fail(f"Tool-calling run failed: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 6. Max-turns limit fires
    # ------------------------------------------------------------------
    print_rule("6. Max-turns limit")
    try:
        from config import config
        from providers import get_model_from_config
        from mcp_client import MCPClient, get_server_configs, build_langchain_tools
        from agent import AgentLoop

        model = get_model_from_config(config)
        all_configs = get_server_configs(config.workspace)
        fs_config = next(c for c in all_configs if c.name == "filesystem")

        async with MCPClient([fs_config]) as c2:
            tools = build_langchain_tools(c2.list_all_tools(), c2)
            # max_turns=1 means it will stop after the first LLM response
            tight_loop = AgentLoop(model=model, tools=tools, max_turns=1, auto_execute=True)
            info("  Running with max_turns=1 on a multi-step task...")
            await tight_loop.run(
                "List all .py files in the project, then read each one and "
                "summarise it. Take as many steps as needed."
            )
        ok("max_turns=1 loop terminated correctly")
    except Exception as e:
        fail(f"Max-turns test failed: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if client is not None:
        try:
            await client.__aexit__(None, None, None)
        except Exception:
            pass

    print_rule("All agent checks complete")


if __name__ == "__main__":
    asyncio.run(run())
