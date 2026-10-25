"""Local inference through Ollama (no rate limits, works offline)."""
from __future__ import annotations

from typing import Callable, Iterator, Sequence

import ollama

from agentkit.config import Settings
from agentkit.llm.base import LLMBackend, Message


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, settings: Settings):
        self.client = ollama.Client(host=settings.ollama_host)
        self.model = settings.ollama_model

    @staticmethod
    def _build(messages: Sequence[Message], system: str) -> list[dict]:
        head = [{"role": "system", "content": system}] if system else []
        return head + [dict(m) for m in messages]

    def chat(self, messages, system="", tools: Sequence[Callable] | None = None) -> str:
        # NOTE: `tools` is ignored for now. The tool loop is added in Step 7.
        resp = self.client.chat(model=self.model, messages=self._build(messages, system))
        return resp.message.content or ""

    def stream(self, messages, system="") -> Iterator[str]:
        for chunk in self.client.chat(
            model=self.model, messages=self._build(messages, system), stream=True
        ):
            if chunk.message.content:
                yield chunk.message.content