"""Central configuration. Only this file reads .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")  # does nothing if .env is missing
load_dotenv(ROOT / ".env.example")

def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    backend: str                      # "ollama" or "gemini"
    ollama_host: str
    ollama_model: str
    embed_model: str
    gemini_api_key: str = field(repr=False)   # hidden when printed
    gemini_model: str
    use_tools: bool
    max_tool_steps: int
    docs_dir: Path
    host: str
    port: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        backend=os.getenv("LLM_BACKEND", "gemini").lower(),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it"),
        use_tools=_bool("USE_TOOLS", True),
        max_tool_steps=int(os.getenv("MAX_TOOL_STEPS", "5")),
        docs_dir=ROOT / os.getenv("DOCS_DIR", "data/docs"),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "7860")),
    )