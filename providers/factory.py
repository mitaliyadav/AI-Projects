"""
providers/factory.py
--------------------
Model-agnostic ChatModel factory.

All providers are returned as a LangChain BaseChatModel so the rest of
the codebase never touches provider-specific classes directly.

Supported providers
-------------------
groq   — ChatGroq   (primary, fast inference, tool-calling support)
openai — ChatOpenAI (secondary, GPT-4o / GPT-4o-mini)
ollama — ChatOllama (local, no API key required)

Usage
-----
    from providers import get_model, get_model_from_config
    from config import config

    model = get_model_from_config(config)
    model_with_tools = model.bind_tools(tools)
    for chunk in model_with_tools.stream(messages):
        ...
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


# ------------------------------------------------------------------
# Supported models per provider (for validation + documentation)
# Add new models here as providers update their offerings.
# ------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, list[str]] = {
    "groq": [
        "llama-3.3-70b-versatile",      # default — best balance of speed/quality
        "llama-3.1-8b-instant",          # fastest, lighter tasks
        "mixtral-8x7b-32768",            # large context window
        "gemma2-9b-it",                  # Google Gemma
    ],
    "openai": [
        "gpt-4o",                        # most capable
        "gpt-4o-mini",                   # cost-effective default
        "gpt-4-turbo",
    ],
    "ollama": [
        "llama3",                        # default local model
        "llama3.1",
        "mistral",
        "codellama",
        "qwen2.5-coder",
        "deepseek-coder-v2",
    ],
}

# Default model to fall back to when none is specified per provider
DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "ollama": "llama3",
}


def get_model(
    provider: str,
    model: str | None = None,
    api_key: str = "",
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Instantiate and return a LangChain ChatModel for the given provider.

    Parameters
    ----------
    provider    : "groq" | "openai" | "ollama"
    model       : Model name. Falls back to DEFAULT_MODELS[provider] if None.
    api_key     : API key string. Not required for Ollama.
    temperature : Sampling temperature. Defaults to 0 for deterministic output.
    **kwargs    : Extra kwargs forwarded to the underlying ChatModel constructor.

    Returns
    -------
    BaseChatModel — supports .stream(), .invoke(), and .bind_tools().

    Raises
    ------
    ValueError  : Unknown provider or model not in supported list.
    ImportError : Required provider package not installed.
    """
    provider = provider.lower().strip()

    if provider not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {list(SUPPORTED_MODELS.keys())}"
        )

    resolved_model = model or DEFAULT_MODELS[provider]

    # Warn (but don't block) if model is not in the known list —
    # providers add new models frequently and we don't want to be too strict.
    known = SUPPORTED_MODELS[provider]
    if resolved_model not in known:
        import warnings
        warnings.warn(
            f"Model '{resolved_model}' is not in the known list for '{provider}'. "
            f"Known models: {known}. Proceeding anyway.",
            stacklevel=2,
        )

    if provider == "groq":
        return _make_groq(resolved_model, api_key, temperature, **kwargs)
    elif provider == "openai":
        return _make_openai(resolved_model, api_key, temperature, **kwargs)
    elif provider == "ollama":
        return _make_ollama(resolved_model, temperature, **kwargs)

    # Should never reach here due to the check above
    raise ValueError(f"Unhandled provider '{provider}'")


# ------------------------------------------------------------------
# Provider-specific constructors (private)
# ------------------------------------------------------------------

def _make_groq(model: str, api_key: str, temperature: float, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise ImportError(
            "langchain-groq is not installed. Run: pip install langchain-groq"
        ) from e

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is required for the Groq provider. "
            "Set it in your .env file or pass it explicitly."
        )

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
        streaming=True,
        **kwargs,
    )


def _make_openai(model: str, api_key: str, temperature: float, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from e

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for the OpenAI provider. "
            "Set it in your .env file or pass it explicitly."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        streaming=True,
        **kwargs,
    )


def _make_ollama(model: str, temperature: float, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from e

    return ChatOllama(
        model=model,
        temperature=temperature,
        **kwargs,
    )


# ------------------------------------------------------------------
# Convenience wrapper — reads directly from config singleton
# ------------------------------------------------------------------

def get_model_from_config(cfg: Any) -> BaseChatModel:
    """
    Build a ChatModel directly from the project Config object.

    Parameters
    ----------
    cfg : config.Config instance

    Returns
    -------
    BaseChatModel ready for tool binding and streaming.
    """
    return get_model(
        provider=cfg.provider,
        model=cfg.model,
        api_key=cfg.get_api_key_for_provider(),
    )
