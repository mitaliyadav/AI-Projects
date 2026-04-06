"""
tests/test_providers.py
-----------------------
Smoke test for the Phase 3 provider factory.
Run with: PYTHONUTF8=1 python tests/test_providers.py
"""

import sys
import time

sys.path.insert(0, ".")

from config import config
from display.console import print_rule, print_error, console
from rich.text import Text
from providers.factory import (
    get_model,
    get_model_from_config,
    SUPPORTED_MODELS,
    DEFAULT_MODELS,
)


def section(title: str) -> None:
    print_rule(title)


def ok(msg: str) -> None:
    console.print(Text(f"  ✔  {msg}", style="bold green"))


def fail(msg: str) -> None:
    console.print(Text(f"  ✘  {msg}", style="bold red"))


def run() -> None:
    section("Phase 3 — Provider Factory Smoke Test")

    # ------------------------------------------------------------------
    # 1. Supported model table
    # ------------------------------------------------------------------
    section("1. Supported models")
    for provider, models in SUPPORTED_MODELS.items():
        default = DEFAULT_MODELS[provider]
        console.print(f"  [bold]{provider}[/bold] (default: [cyan]{default}[/cyan])")
        for m in models:
            marker = " ◀ default" if m == default else ""
            console.print(f"    - {m}{marker}")
    ok("SUPPORTED_MODELS and DEFAULT_MODELS loaded")

    # ------------------------------------------------------------------
    # 2. Unknown provider raises ValueError
    # ------------------------------------------------------------------
    section("2. Unknown provider raises ValueError")
    try:
        get_model("anthropic")
        fail("Should have raised ValueError")
    except ValueError as e:
        ok(f"ValueError raised as expected: {e}")

    # ------------------------------------------------------------------
    # 3. Missing API key raises ValueError
    # ------------------------------------------------------------------
    section("3. Missing API key raises ValueError")
    try:
        get_model("groq", api_key="")
        fail("Should have raised ValueError for empty key")
    except ValueError as e:
        ok(f"ValueError raised as expected: {e}")

    # ------------------------------------------------------------------
    # 4. Correct LangChain class is returned per provider
    # ------------------------------------------------------------------
    section("4. Correct class returned per provider")

    # Groq
    try:
        from langchain_groq import ChatGroq
        model = get_model("groq", api_key=config.groq_api_key)
        assert isinstance(model, ChatGroq), f"Expected ChatGroq, got {type(model)}"
        ok(f"groq → {type(model).__name__}")
    except Exception as e:
        fail(f"groq: {e}")

    # OpenAI
    try:
        from langchain_openai import ChatOpenAI
        model = get_model("openai", api_key=config.openai_api_key)
        assert isinstance(model, ChatOpenAI), f"Expected ChatOpenAI, got {type(model)}"
        ok(f"openai → {type(model).__name__}")
    except Exception as e:
        fail(f"openai: {e}")

    # Ollama (no key needed)
    try:
        from langchain_ollama import ChatOllama
        model = get_model("ollama")
        assert isinstance(model, ChatOllama), f"Expected ChatOllama, got {type(model)}"
        ok(f"ollama → {type(model).__name__}")
    except Exception as e:
        fail(f"ollama: {e}")

    # ------------------------------------------------------------------
    # 5. get_model_from_config uses the config singleton
    # ------------------------------------------------------------------
    section("5. get_model_from_config reads from config singleton")
    try:
        model = get_model_from_config(config)
        ok(
            f"get_model_from_config → {type(model).__name__} "
            f"(provider={config.provider}, model={config.model})"
        )
    except Exception as e:
        fail(f"get_model_from_config: {e}")

    # ------------------------------------------------------------------
    # 6. bind_tools returns a runnable (no actual tools needed)
    # ------------------------------------------------------------------
    section("6. bind_tools returns a Runnable")
    try:
        model = get_model("groq", api_key=config.groq_api_key)
        bound = model.bind_tools([])   # empty tool list — just checks the interface
        ok(f"bind_tools([]) → {type(bound).__name__}")
    except Exception as e:
        fail(f"bind_tools: {e}")

    # ------------------------------------------------------------------
    # 7. Real streaming call to Groq (primary provider)
    # ------------------------------------------------------------------
    section("7. Live streaming call to Groq")
    try:
        from langchain_core.messages import HumanMessage
        model = get_model("groq", api_key=config.groq_api_key)
        prompt = [HumanMessage(content="Reply with exactly three words: 'provider test passed'")]

        console.print("  Streaming response: ", end="")
        accumulated = []
        for chunk in model.stream(prompt):
            token = chunk.content
            accumulated.append(token)
            console.print(token, end="", highlight=False)

        console.print()  # newline after stream
        full_response = "".join(accumulated).strip().lower()
        ok(f"Stream complete. Response: '{full_response}'")
    except Exception as e:
        fail(f"Groq streaming call failed: {e}")

    section("All provider checks complete")


if __name__ == "__main__":
    run()
