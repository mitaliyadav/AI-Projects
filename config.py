"""
config.py
---------
Loads environment variables from .env and exposes a typed Config object.
All other modules import `config` from here — never read os.environ directly.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Central configuration object.
    Values are loaded from environment variables or a .env file.
    CLI flags in main.py override these at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # API Keys
    # ------------------------------------------------------------------
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    context7_api_key: str = Field(default="", alias="CONTEXT7_API_KEY")

    # ------------------------------------------------------------------
    # Provider & Model
    # ------------------------------------------------------------------
    provider: str = Field(default="groq", alias="PROVIDER")
    model: str = Field(default="llama-3.3-70b-versatile", alias="MODEL")

    # ------------------------------------------------------------------
    # Agent Behaviour
    # ------------------------------------------------------------------
    max_turns: int = Field(default=20, alias="MAX_TURNS")
    auto_execute: bool = Field(default=False, alias="AUTO_EXECUTE")

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------
    workspace: str = Field(default="", alias="WORKSPACE")

    # ------------------------------------------------------------------
    # Internal paths (not from .env)
    # ------------------------------------------------------------------
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"groq", "openai", "ollama"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}, got '{v}'")
        return v

    @field_validator("max_turns")
    @classmethod
    def validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_turns must be at least 1")
        if v > 100:
            raise ValueError("max_turns must be 100 or fewer to avoid runaway costs")
        return v

    @field_validator("workspace")
    @classmethod
    def resolve_workspace(cls, v: str) -> str:
        """Resolve workspace to an absolute path. Defaults to cwd."""
        if not v or v.strip() == "":
            return str(Path.cwd())
        resolved = Path(v).resolve()
        if not resolved.exists():
            raise ValueError(f"Workspace path does not exist: {resolved}")
        return str(resolved)

    def get_api_key_for_provider(self) -> str:
        """Return the API key for the currently configured provider."""
        mapping = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "ollama": "",  # Ollama is local, no key needed
        }
        return mapping.get(self.provider, "")

    @property
    def chroma_db_path(self) -> Path:
        """Absolute path to the persisted ChromaDB directory."""
        return self.project_root / "mcp_servers" / "rag_server" / "chroma_db"

    def check_node(self) -> bool:
        """Return True if Node.js and npx are available on PATH."""
        return shutil.which("node") is not None and shutil.which("npx") is not None

    def check_ollama(self) -> bool:
        """Return True if the ollama binary is available on PATH."""
        return shutil.which("ollama") is not None

    def validate_environment(self) -> list[str]:
        """
        Run pre-flight checks and return a list of warning strings.
        Does NOT raise — callers decide how to surface warnings.
        """
        warnings: list[str] = []

        # Provider API key check
        key = self.get_api_key_for_provider()
        if self.provider != "ollama" and not key:
            warnings.append(
                f"No API key found for provider '{self.provider}'. "
                f"Set {self.provider.upper()}_API_KEY in your .env file."
            )

        # Node / npx check (required for filesystem + context7 MCP servers)
        if not self.check_node():
            warnings.append(
                "Node.js / npx not found on PATH. "
                "Install Node.js from https://nodejs.org — "
                "it is required for the filesystem and Context7 MCP servers."
            )

        # Ollama check (required for local embeddings in RAG server)
        if not self.check_ollama():
            warnings.append(
                "Ollama not found on PATH. "
                "Install from https://ollama.com — "
                "it is required for local embeddings in the RAG server."
            )

        return warnings


# ------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ------------------------------------------------------------------
config = Config()
