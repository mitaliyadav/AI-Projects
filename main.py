"""
main.py
--------
CLI entry point for the AI coding assistant.

Usage
-----
  python main.py                        # interactive REPL
  python main.py "Fix the login bug"    # single task, then exit
  python main.py --setup-rag            # build RAG index and exit
  python main.py --auto                 # skip per-tool confirmation
  python main.py --max-turns 10         # tighter turn budget
  python main.py --provider openai      # use OpenAI instead of Groq
  python main.py --model gpt-4o         # override the model

After installation (pip install -e .):
  assistant                             # same as python main.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.prompt import Prompt

from config import config
from display.console import (
    console,
    print_error,
    print_rule,
    print_status,
    print_welcome_banner,
)

app = typer.Typer(
    name="assistant",
    help="Autonomous AI coding assistant powered by LangChain + MCP.",
    add_completion=False,
)

# Special REPL commands
_REPL_EXIT_CMDS = {"exit", "quit", "q", ":q"}
_REPL_CLEAR_CMD = "clear"
_REPL_HELP_CMD = "help"

_REPL_HELP_TEXT = """\
[bold cyan]REPL commands[/bold cyan]
  [dim]<task>[/dim]   Describe a coding task and press Enter
  [dim]clear[/dim]    Clear conversation history (start fresh)
  [dim]exit[/dim]     Quit the assistant  (also: quit, q, Ctrl+C)
  [dim]help[/dim]     Show this message
"""


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

@app.command()
def main(
    task: Optional[str] = typer.Argument(
        None,
        help="Task to run immediately (omit for interactive REPL mode).",
    ),
    auto: bool = typer.Option(
        False,
        "--auto",
        help="Auto-execute tools without asking for confirmation.",
    ),
    max_turns: Optional[int] = typer.Option(
        None,
        "--max-turns",
        min=1,
        max=100,
        help="Hard limit on agentic loop iterations (default: from .env or 20).",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="LLM provider: groq | openai | ollama  (default: from .env or groq).",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name override (e.g. gpt-4o, llama-3.3-70b-versatile).",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace",
        help="Workspace directory the filesystem server may access (default: cwd).",
    ),
    setup_rag: bool = typer.Option(
        False,
        "--setup-rag",
        help="Build (or rebuild) the LangChain documentation index and exit.",
    ),
) -> None:
    """Autonomous AI coding assistant — reads, edits, and executes code for you."""

    # ------------------------------------------------------------------
    # --setup-rag: build index and exit
    # ------------------------------------------------------------------
    if setup_rag:
        _run_setup_rag()
        return

    # ------------------------------------------------------------------
    # Resolve effective config (CLI flags override .env values)
    # ------------------------------------------------------------------
    effective_provider = provider or config.provider
    effective_model = model_name  # None → factory picks the default for the provider
    effective_max_turns = max_turns if max_turns is not None else config.max_turns
    effective_auto = auto or config.auto_execute
    effective_workspace = workspace or str(config.workspace)

    # ------------------------------------------------------------------
    # Run the async session
    # ------------------------------------------------------------------
    try:
        asyncio.run(
            _async_session(
                task=task,
                provider=effective_provider,
                model_name=effective_model,
                max_turns=effective_max_turns,
                auto_execute=effective_auto,
                workspace=effective_workspace,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted — goodbye.[/dim]")
    except Exception as exc:
        print_error(f"Fatal error: {exc}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Async session — everything after arg parsing
# ---------------------------------------------------------------------------

async def _async_session(
    task: Optional[str],
    provider: str,
    model_name: Optional[str],
    max_turns: int,
    auto_execute: bool,
    workspace: str,
) -> None:
    """Connect to MCP servers, build tools, and run one task or a full REPL."""

    from providers import get_model
    from mcp_client import MCPClient, get_server_configs, build_langchain_tools
    from agent import AgentLoop

    # ------------------------------------------------------------------
    # Resolve model
    # ------------------------------------------------------------------
    _key_map = {
        "groq": config.groq_api_key,
        "openai": config.openai_api_key,
        "ollama": "",
    }
    api_key = _key_map.get(provider, "")
    model = get_model(provider=provider, model=model_name, api_key=api_key)

    # Determine the display model name (may be None → factory chose default)
    display_model = model_name or _default_model_name(provider)

    # ------------------------------------------------------------------
    # Welcome banner
    # ------------------------------------------------------------------
    print_welcome_banner(
        provider=provider,
        model=display_model,
        max_turns=max_turns,
        auto_execute=auto_execute,
    )

    # ------------------------------------------------------------------
    # Pre-flight warnings (don't abort — partial toolset is better than nothing)
    # ------------------------------------------------------------------
    warnings = config.validate_environment()
    for w in warnings:
        console.print(f"  [yellow]⚠[/yellow]  {w}")

    # ------------------------------------------------------------------
    # Connect to MCP servers
    # ------------------------------------------------------------------
    server_configs = get_server_configs(workspace)
    mcp_client = MCPClient(server_configs)

    with print_status("Connecting…"):
        try:
            await mcp_client.__aenter__()
        except Exception as exc:
            print_error(f"Failed to start MCP servers: {exc}")
            raise typer.Exit(code=1)

    try:
        # Build LangChain-compatible tools
        mcp_tools = mcp_client.list_all_tools()
        lc_tools = build_langchain_tools(mcp_tools, mcp_client)

        # Filter out redundant / low-value tools.
        # Groq's llama models fail tool-call generation above ~12–14 tools.
        # Several filesystem tools added in recent server versions duplicate
        # functionality that simpler alternatives already cover.
        _SKIP_TOOLS: set[str] = {
            "read_text_file",          # duplicates read_file
            "read_media_file",         # LLM cannot interpret binary media
            "read_multiple_files",     # model can call read_file in sequence
            "list_directory_with_sizes",  # duplicates list_directory
            "get_file_info",           # covered by list_directory
            "get_rag_status",          # maintenance/health tool; not task-relevant
        }
        lc_tools = [t for t in lc_tools if t.name not in _SKIP_TOOLS]

        # Build the agentic loop
        agent = AgentLoop(
            model=model,
            tools=lc_tools,
            max_turns=max_turns,
            auto_execute=auto_execute,
        )

        # ------------------------------------------------------------------
        # Single-task mode (task provided on command line)
        # ------------------------------------------------------------------
        if task:
            await agent.run(task)
            return

        # ------------------------------------------------------------------
        # Interactive REPL mode
        # ------------------------------------------------------------------
        console.print(
            "\n[dim]Type a task and press Enter.  "
            "'help' for commands, 'exit' to quit.[/dim]\n"
        )

        while True:
            try:
                user_input = console.input("[bold green]>[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in _REPL_EXIT_CMDS:
                console.print("[dim]Goodbye.[/dim]")
                break

            if user_input.lower() == _REPL_CLEAR_CMD:
                agent.clear_history()
                print_rule("History cleared")
                continue

            if user_input.lower() == _REPL_HELP_CMD:
                console.print(_REPL_HELP_TEXT)
                continue

            # Run the task
            try:
                await agent.run(user_input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Task interrupted.[/yellow]")
                agent.clear_history()
            except Exception as exc:
                print_error(f"Task failed: {exc}")
                agent.clear_history()

            # Blank line between tasks for readability
            console.print()

    finally:
        # Always shut down MCP subprocesses cleanly
        with print_status("Shutting down MCP servers…"):
            try:
                await mcp_client.__aexit__(None, None, None)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# --setup-rag helper (synchronous — indexer does its own asyncio if needed)
# ---------------------------------------------------------------------------

def _run_setup_rag() -> None:
    """Run the RAG indexer and exit."""
    console.print("[bold cyan]Setting up LangChain documentation index…[/bold cyan]\n")
    try:
        from mcp_servers.rag_server.indexer import build_index
        build_index(force_rebuild=False)
    except Exception as exc:
        print_error(f"RAG setup failed: {exc}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _default_model_name(provider: str) -> str:
    """Return the default model name for a provider (for display only)."""
    from providers import DEFAULT_MODELS
    return DEFAULT_MODELS.get(provider, "unknown")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
