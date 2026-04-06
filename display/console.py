"""
display/console.py
------------------
All terminal output is routed through this module.
Uses Rich for styled, visually appealing output.

Public API
----------
print_welcome_banner(provider, model, max_turns, auto_execute)
print_user_message(text)
print_assistant_stream(chunks)           -> str  (accumulated text)
print_tool_call(name, args)
print_tool_result(name, result, success)
print_confirmation_prompt(name, args)    -> bool (True = confirmed)
print_status(msg)                        context manager with spinner
print_turn_warning(current, max_turns)
print_turn_limit_reached(max_turns)
print_error(msg)
print_rule(title)
print_task_complete()
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import Generator, Iterable

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.prompt import Confirm
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme
from rich.panel import Panel

# ------------------------------------------------------------------
# Windows UTF-8 fix
# Force stdout/stderr to UTF-8 so emoji and box-drawing characters
# render correctly in Windows Terminal and VS Code terminals.
# ------------------------------------------------------------------
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------
_THEME = Theme(
    {
        "banner.title": "bold bright_cyan",
        "banner.subtitle": "dim cyan",
        "user.label": "bold cyan",
        "user.text": "white",
        "assistant.label": "bold bright_green",
        "assistant.text": "bright_white",
        "tool.header": "bold yellow",
        "tool.border_pending": "yellow",
        "tool.border_ok": "green",
        "tool.border_err": "red",
        "tool.key": "dim yellow",
        "confirm.prompt": "bold magenta",
        "warning.text": "bold yellow",
        "error.text": "bold red",
        "turn.badge": "bold magenta",
        "status.text": "dim cyan",
        "rule.style": "dim white",
        "complete.text": "bold bright_green",
    }
)

# Module-level console — import this in other modules when you need
# direct Rich access (e.g. for Progress bars in future phases).
# force_terminal=True ensures Rich renders ANSI styles even when stdout
# is not a TTY (e.g. piped output, some CI environments).
console = Console(theme=_THEME, highlight=False, force_terminal=True)


# ------------------------------------------------------------------
# Welcome banner
# ------------------------------------------------------------------
def print_welcome_banner(
    provider: str,
    model: str,
    max_turns: int,
    auto_execute: bool,
) -> None:
    """Print the startup banner with session configuration."""
    title = Text("Bibble - Your AI Assistant", style="banner.title")
    subtitle = Text("Autonomous. Agentic. Local-first.", style="banner.subtitle")

    config_lines = Text()
    config_lines.append("  Provider   : ", style="tool.key")
    config_lines.append(f"{provider}\n", style="bold white")
    config_lines.append("  Model      : ", style="tool.key")
    config_lines.append(f"{model}\n", style="bold white")
    config_lines.append("  Auto-exec  : ", style="tool.key")
    auto_val = "ON" if auto_execute else "OFF"
    config_lines.append(auto_val, style="bold white")

    body = Text.assemble(title, "\n", subtitle, "\n\n", config_lines)

    console.print(
        Panel(
            body,
            border_style="bright_cyan",
            padding=(1, 3),
            subtitle="[dim]Type your task below · Ctrl+C to quit[/dim]",
        )
    )
    console.print()


# ------------------------------------------------------------------
# User message
# ------------------------------------------------------------------
def print_user_message(text: str) -> None:
    """Echo the user's input with a styled label."""
    label = Text("You › ", style="user.label")
    content = Text(text, style="user.text")
    console.print(Text.assemble(label, content))
    console.print()


# ------------------------------------------------------------------
# Assistant streaming
# ------------------------------------------------------------------
def print_assistant_stream(chunks: Iterable[str]) -> str:
    """
    Stream LLM response tokens to the terminal in real-time.

    Accepts an iterable of string chunks (e.g. from LangChain .stream()).
    Returns the full accumulated response text.
    """
    label = Text("Assistant › ", style="assistant.label")
    accumulated: list[str] = []

    with Live(console=console, refresh_per_second=15, vertical_overflow="visible") as live:
        for chunk in chunks:
            accumulated.append(chunk)
            full_text = "".join(accumulated)
            display = Text.assemble(label, Text(full_text, style="assistant.text"))
            live.update(display)

    console.print()  # blank line after streaming completes
    return "".join(accumulated)


# ------------------------------------------------------------------
# Tool call display
# ------------------------------------------------------------------
def print_tool_call(name: str, args: dict) -> None:
    """Display an outgoing tool call in a styled yellow panel."""
    header = Text()
    header.append("Tool  : ", style="tool.key")
    header.append(f"{name}\n", style="tool.header")
    header.append("Args  :", style="tool.key")

    args_str = json.dumps(args, indent=2, ensure_ascii=False)
    args_display = Syntax(args_str, "json", theme="monokai", word_wrap=True)

    console.print(
        Panel(
            Group(header, args_display),
            border_style="tool.border_pending",
            padding=(0, 1),
            title="[tool.header]🔧 Tool Call[/tool.header]",
            title_align="left",
        )
    )


# ------------------------------------------------------------------
# Tool result display
# ------------------------------------------------------------------
def print_tool_result(name: str, result: str, success: bool = True) -> None:
    """Display a tool result in a green (success) or red (error) panel."""
    icon = "✅" if success else "❌"
    border = "tool.border_ok" if success else "tool.border_err"
    label_style = "assistant.label" if success else "error.text"

    header = Text()
    header.append(f"{icon} Result from: ", style="tool.key")
    header.append(f"{name}\n\n", style=label_style)

    # Truncate very long results in the display (full result still passed to LLM)
    display_result = (
        result if len(result) <= 2000 else result[:2000] + "\n\n… [truncated for display]"
    )
    body = Text.assemble(header, Text(display_result, style="dim white"))

    console.print(
        Panel(
            body,
            border_style=border,
            padding=(0, 1),
            title="[dim]Tool Result[/dim]",
            title_align="left",
        )
    )
    console.print()


# ------------------------------------------------------------------
# Confirmation prompt
# ------------------------------------------------------------------
def print_confirmation_prompt(name: str, args: dict) -> bool:
    """
    Ask the user whether to execute a tool call.
    Returns True if the user confirms, False if they decline.
    """
    args_str = json.dumps(args, indent=2, ensure_ascii=False)
    args_display = Syntax(args_str, "json", theme="monokai", word_wrap=True)

    header = Text()
    header.append("Tool  : ", style="tool.key")
    header.append(f"{name}\n", style="bold white")
    header.append("Args  :\n", style="tool.key")

    console.print(
        Panel(
            Group(header, args_display),
            border_style="magenta",
            padding=(0, 1),
            title="[confirm.prompt]⚠️  Confirm Tool Execution[/confirm.prompt]",
            title_align="left",
        )
    )

    return Confirm.ask("[confirm.prompt]Execute this tool?[/confirm.prompt]", console=console)


# ------------------------------------------------------------------
# Spinner / status context manager
# ------------------------------------------------------------------
@contextmanager
def print_status(msg: str) -> Generator[None, None, None]:
    """
    Context manager that shows a spinner while an operation runs.

    Usage:
        with print_status("Connecting to MCP servers..."):
            do_work()
    """
    spinner = Spinner("dots", text=Text(f" {msg}", style="status.text"))
    with Live(spinner, console=console, refresh_per_second=10):
        yield


# ------------------------------------------------------------------
# Turn warnings
# ------------------------------------------------------------------
def print_turn_warning(current: int, max_turns: int) -> None:
    """Warn when nearing the max turn limit (fires at 80% threshold)."""
    remaining = max_turns - current
    msg = Text()
    msg.append("⚠️  Turn warning: ", style="warning.text")
    msg.append(
        f"Turn {current}/{max_turns} — {remaining} turn(s) remaining before auto-exit.",
        style="bold white",
    )
    console.print(msg)
    console.print()


def print_turn_limit_reached(max_turns: int) -> None:
    """Inform the user the hard turn limit was hit and the loop has exited."""
    console.print(
        Panel(
            Text(
                f"Maximum turn limit of {max_turns} reached.\n"
                "The agent has stopped to prevent runaway costs.\n"
                "You can continue by submitting a new task.",
                style="warning.text",
            ),
            border_style="yellow",
            title="[warning.text]🛑 Turn Limit Reached[/warning.text]",
            title_align="left",
            padding=(0, 1),
        )
    )
    console.print()


# ------------------------------------------------------------------
# Error display
# ------------------------------------------------------------------
def print_error(msg: str) -> None:
    """Display a red error panel."""
    console.print(
        Panel(
            Text(f"{msg}", style="error.text"),
            border_style="red",
            title="[error.text]❌ Error[/error.text]",
            title_align="left",
            padding=(0, 1),
        )
    )
    console.print()


# ------------------------------------------------------------------
# Divider rule
# ------------------------------------------------------------------
def print_rule(title: str = "") -> None:
    """Print a horizontal divider rule, optionally with a title."""
    console.print(Rule(title=title, style="rule.style"))


# ------------------------------------------------------------------
# Task complete
# ------------------------------------------------------------------
def print_task_complete() -> None:
    """Print a task-complete confirmation after the agentic loop exits cleanly."""
    console.print(Text("✔  Task complete.", style="complete.text"))
    console.print()
