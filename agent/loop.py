"""
agent/loop.py
--------------
Core agentic loop: streams the LLM response, detects tool calls, asks for
confirmation if needed, executes tools, feeds results back, and repeats
until the model stops calling tools or the max-turns limit is reached.

Usage
-----
    from agent.loop import AgentLoop

    loop = AgentLoop(
        model=model,
        tools=tools,
        max_turns=20,
        auto_execute=False,
    )
    await loop.run("Refactor the login function in auth.py")
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from agent.history import MessageHistory
from display.console import (
    console,
    print_confirmation_prompt,
    print_error,
    print_rule,
    print_status,
    print_task_complete,
    print_tool_call,
    print_tool_result,
    print_turn_limit_reached,
    print_turn_warning,
)
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an autonomous AI coding assistant. You help developers read, \
understand, and modify code to complete tasks fully and correctly.

## Your tools
- **Filesystem tools** — read, write, edit, list, search files in the workspace
- **Context7** — fetch up-to-date library documentation from the web
- **search_langchain_docs** — semantic search over indexed LangChain docs; use ONLY \
when you need to look up LangChain API syntax, classes, or patterns — NOT to answer \
questions about the current project

## How to work
1. Start by exploring the relevant files so you understand the codebase
2. Make changes in small, verifiable steps
3. After editing a file, read it back to confirm the change is correct
4. When the task is done, give a clear summary of everything you changed

## Important rules
- Never repeat a failed approach — diagnose the error and try differently
- Do not ask clarifying questions unless the task is genuinely ambiguous
- Stop and explain clearly if you truly cannot complete the task
- Prefer targeted edits over full file rewrites
- Only call tools that are explicitly listed above. Never attempt to call \
tools you know from training (e.g. brave_search, web_search) if they are \
not in your available tool list — instead, explain what you cannot do.
"""

# Warn when this fraction of max_turns has been used
_WARN_FRACTION = 0.75


class AgentLoop:
    """
    Drives one task through multiple LLM turns until completion or the
    max-turns limit is reached.

    Parameters
    ----------
    model        : A LangChain BaseChatModel (Groq / OpenAI / Ollama).
    tools        : List of LangChain BaseTools (from MCP + tool adapter).
    max_turns    : Hard upper bound on the number of turns.
    auto_execute : If True, execute tools without a confirmation prompt.
    """

    def __init__(
        self,
        model: Any,
        tools: list[BaseTool],
        max_turns: int = 20,
        auto_execute: bool = False,
    ) -> None:
        self._model = model
        self._tools = tools
        self._max_turns = max_turns
        self._auto_execute = auto_execute
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in tools}
        self._history = MessageHistory(system_prompt=SYSTEM_PROMPT)
        self._oriented = False  # True after the first workspace scan

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: str) -> None:
        """
        Run the agentic loop for the given task.

        Adds the task as a HumanMessage, then iterates through turns until
        the model produces a response with no tool calls (meaning it is
        done) or the max-turns limit is hit.
        """
        # On the very first task, do a lightweight workspace scan so Bibble
        # has a structural map of the project before answering anything.
        if not self._oriented:
            with print_status(
                "Bibble is just walking around to take a look! "
                "Please give him a moment…"
            ):
                await self._orient_workspace()
            self._oriented = True

        self._history.add_human(task)

        for turn in range(1, self._max_turns + 1):
            # --- warn when approaching the limit ---
            warn_at = max(1, int(self._max_turns * _WARN_FRACTION))
            if turn == warn_at:
                print_turn_warning(turn, self._max_turns)

            # --- stream the LLM response for this turn ---
            try:
                ai_message = await self._stream_turn(self._history.get_messages())
            except Exception as exc:
                print_error(f"LLM error on turn {turn}: {exc}")
                raise  # let the caller (REPL) handle it cleanly
            self._history.add_ai(ai_message)

            # --- no tool calls → model is done ---
            if not ai_message.tool_calls:
                print_task_complete()
                break

            # --- execute each tool call ---
            any_executed = await self._execute_tool_calls(ai_message.tool_calls)

            # If the user declined every tool call, stop to avoid an
            # infinite loop where the model keeps requesting the same tools.
            if not any_executed:
                print_error(
                    "All tool calls were declined. Stopping to avoid a loop."
                )
                break

        else:
            # Loop exhausted — turn limit reached
            print_turn_limit_reached(self._max_turns)

    def clear_history(self) -> None:
        """Reset conversation history (for a new task session)."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Private — workspace orientation
    # ------------------------------------------------------------------

    async def _orient_workspace(self) -> None:
        """
        List the top-level workspace directory and inject the result into
        history as a pre-populated exchange so the model starts every session
        with a structural map of the codebase.

        We use list_directory (top-level only) rather than directory_tree
        (fully recursive) to avoid blowing the LLM's context window on
        projects that contain large generated directories such as .venv,
        __pycache__, or chroma_db.  Bibble reads deeper files on demand
        through read_file / list_directory tool calls during the conversation.

        This is best-effort: if the tool is unavailable or errors, we skip
        silently rather than blocking the user's first task.
        """
        tool = self._tool_map.get("list_directory")
        if tool is None:
            return
        try:
            listing = await tool.arun({"path": "."})
        except Exception:
            return  # orientation failed — proceed without it

        # Safety cap: even a shallow listing can be long on very wide repos
        _MAX_CHARS = 3_000
        if len(listing) > _MAX_CHARS:
            listing = listing[:_MAX_CHARS] + "\n… [listing truncated]"

        #print(f"[DEBUG orient] listing OK — {len(listing)} chars", flush=True)

        # Append to the system prompt rather than injecting fake conversation
        # turns.  Fake HumanMessage/AIMessage pairs in history confuse Groq's
        # function-call JSON generation even when total token count is small.
        self._history.append_to_system_prompt(
            f"## Workspace — top-level structure\n{listing}"
        )

        sys_len = len(self._history.get_messages()[0].content)
        #print(f"[DEBUG orient] system prompt now {sys_len} chars", flush=True)

    # ------------------------------------------------------------------
    # Private — streaming
    # ------------------------------------------------------------------

    async def _stream_turn(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Stream one LLM turn.

        Displays tokens in-place with Rich Live as they arrive, then returns
        the fully accumulated AIMessage (including any tool_calls).
        """
        bound = self._model.bind_tools(self._tools, parallel_tool_calls=False)

        # DEBUG — inspect what we're sending to the model
        total_chars = sum(
            len(m.content) if isinstance(m.content, str)
            else sum(p.get("text", "") and len(p.get("text", "")) for p in m.content if isinstance(p, dict))
            for m in messages
        )
        #print(
        #    f"[DEBUG stream] msgs={len(messages)} total_chars={total_chars} tools={len(self._tools)}",
        #    flush=True,
        #)
        

        accumulated: AIMessage | None = None
        text_buf = ""

        try:
            with Live("", console=console, refresh_per_second=15, transient=False) as live:
                async for chunk in bound.astream(messages):
                    # --- accumulate text for display ---
                    if isinstance(chunk.content, str):
                        text_buf += chunk.content
                    elif isinstance(chunk.content, list):
                        for part in chunk.content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_buf += part.get("text", "")

                    # --- update the live display ---
                    if text_buf:
                        live.update(
                            Text.assemble(
                                Text("Assistant  ", style="bold cyan"),
                                Text(text_buf, style="default"),
                            )
                        )

                    # --- accumulate the full message (text + tool_calls) ---
                    if accumulated is None:
                        accumulated = chunk  # type: ignore[assignment]
                    else:
                        accumulated = accumulated + chunk  # type: ignore[operator]
        except Exception as exc:
            # DEBUG — walk the exception chain and dump failed_generation
            cur: BaseException | None = exc
            failed_gen: str | None = None
            while cur is not None:
                #print(f"[DEBUG exc] {type(cur).__name__}: {cur}", flush=True)
                if hasattr(cur, "body") and cur.body:
                    #print(f"[DEBUG exc body] {cur.body}", flush=True)
                    body = cur.body if isinstance(cur.body, dict) else {}
                    if body.get("code") == "tool_use_failed" and "failed_generation" in body:
                        failed_gen = body["failed_generation"]
                if hasattr(cur, "response") and cur.response:
                    try:
                        print("")
                    except Exception:
                        pass
                cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)

            # Recovery: the model sometimes emits Hermes-format tool calls
            # (<function=name{...}</function>) instead of standard JSON.
            # Parse and convert rather than crashing.
            if failed_gen is not None:
                recovered = self._parse_hermes_tool_call(failed_gen)
                if recovered is not None:
                    #print(
                    #    f"[DEBUG] Recovered Hermes tool call: {recovered.tool_calls}",
                    #    flush=True,
                    #)
                    return recovered  # type: ignore[return-value]

            raise

        # Ensure a clean newline after the live output
        if text_buf:
            console.print()

        # Safety: should not happen, but avoid returning None
        if accumulated is None:
            accumulated = AIMessage(content="")

        return accumulated  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private — tool execution
    # ------------------------------------------------------------------

    # Tools that modify the workspace — these always require user confirmation.
    # Everything else (reads, searches, doc lookups) executes automatically.
    _WRITE_TOOLS: frozenset[str] = frozenset({
        "write_file",
        "edit_file",
        "create_directory",
        "move_file",
    })

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> bool:
        """
        Execute all tool calls in the current turn.

        Write operations (write_file, edit_file, create_directory, move_file)
        require explicit user confirmation unless --auto is set.
        Read-only tools execute immediately without prompting.

        Returns True if at least one tool was executed, False if all were
        declined by the user.
        """
        any_executed = False

        for tool_call in tool_calls:
            name: str = tool_call.get("name", "")
            args: dict = tool_call.get("args", {})
            tool_call_id: str = tool_call.get("id", "")

            # --- display what the model wants to call ---
            print_tool_call(name, args)

            # --- confirmation prompt for write operations only ---
            needs_confirm = (not self._auto_execute) and (name in self._WRITE_TOOLS)
            if needs_confirm:
                confirmed = print_confirmation_prompt(name, args)
                if not confirmed:
                    # Record the refusal in history so the model knows
                    self._history.add_tool_result(
                        tool_call_id=tool_call_id,
                        name=name,
                        content="[User declined this tool call]",
                    )
                    continue

            # --- find and run the tool ---
            tool = self._tool_map.get(name)
            if tool is None:
                result = f"Error: tool '{name}' not found."
                success = False
            else:
                try:
                    result = await tool.arun(args)
                    success = True
                except Exception as exc:
                    result = f"Tool execution error: {exc}"
                    success = False

            # --- display the result ---
            print_tool_result(name, result, success)

            # --- add to history ---
            self._history.add_tool_result(
                tool_call_id=tool_call_id,
                name=name,
                content=result,
            )
            any_executed = True

        return any_executed

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hermes_tool_call(failed_generation: str) -> AIMessage | None:
        """
        Some LLaMA models emit tool calls in Hermes XML format instead of JSON:
            <function=tool_name{"arg": "val"}</function>

        Parse that format and return a valid AIMessage so the loop can continue
        instead of crashing with 'Failed to call a function'.

        Returns None if the format doesn't match or JSON parsing fails.
        """
        match = re.search(
            r"<function=(?P<name>[a-zA-Z0-9_\-]+)(?P<args>\{.*?\})</function>",
            failed_generation,
            re.DOTALL,
        )
        if not match:
            return None
        name = match.group("name")
        try:
            args = json.loads(match.group("args"))
        except json.JSONDecodeError:
            return None
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }
            ],
        )

    @staticmethod
    def _pretty_args(args: dict) -> str:
        """Return args as a compact JSON string for display."""
        try:
            return json.dumps(args, ensure_ascii=False, indent=2)
        except Exception:
            return str(args)
