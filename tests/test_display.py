"""
tests/test_display.py
---------------------
Smoke test for the Phase 2 display layer.
Run with: python tests/test_display.py
"""

import sys
import time

sys.path.insert(0, ".")

from display.console import (
    print_welcome_banner,
    print_user_message,
    print_assistant_stream,
    print_tool_call,
    print_tool_result,
    print_confirmation_prompt,
    print_status,
    print_turn_warning,
    print_turn_limit_reached,
    print_error,
    print_rule,
    print_task_complete,
)


def fake_stream(text: str):
    """Simulate chunked LLM streaming token by token."""
    for word in text.split():
        yield word + " "
        time.sleep(0.05)


def run():
    print_rule("Phase 2 — Display Layer Smoke Test")

    # 1. Welcome banner
    print_welcome_banner(
        provider="groq",
        model="llama-3.3-70b-versatile",
        max_turns=20,
        auto_execute=False,
    )

    # 2. User message
    print_user_message("Write a Python function that reads a JSON file and returns its contents.")

    # 3. Assistant streaming
    response = print_assistant_stream(
        fake_stream(
            "Sure! I will read the file using the filesystem tool first, "
            "then write the function based on what I find."
        )
    )
    print(f"[smoke test] Accumulated text length: {len(response)} chars")

    print_rule()

    # 4. Tool call
    print_tool_call(
        name="read_file",
        args={"path": "/workspace/data/config.json"},
    )

    # 5. Tool result — success
    print_tool_result(
        name="read_file",
        result='{\n  "host": "localhost",\n  "port": 8080,\n  "debug": true\n}',
        success=True,
    )

    # 6. Tool result — failure
    print_tool_result(
        name="read_file",
        result="FileNotFoundError: /workspace/data/config.json does not exist.",
        success=False,
    )

    # 7. Turn warning
    print_turn_warning(current=16, max_turns=20)

    # 8. Turn limit reached
    print_turn_limit_reached(max_turns=20)

    # 9. Error panel
    print_error("Failed to connect to MCP server: connection refused on stdio transport.")

    # 10. Status spinner
    with print_status("Connecting to MCP servers..."):
        time.sleep(2)

    # 11. Task complete
    print_task_complete()

    print_rule("All display components rendered successfully")


if __name__ == "__main__":
    run()
