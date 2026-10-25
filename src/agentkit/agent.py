"""Glue layer: turns chat history + a new message into a reply."""
from __future__ import annotations

from typing import Callable, Iterator, Sequence

from agentkit.config import get_settings
from agentkit.llm import LLMBackend, Message, get_backend
from agentkit.prompts import SYSTEM_PROMPT


class Agent:
    def __init__(
        self,
        backend: LLMBackend | None = None,
        tools: Sequence[Callable] | None = None,
        system: str = SYSTEM_PROMPT,
    ):
        self.backend = backend or get_backend(get_settings())
        self.tools = list(tools) if tools else []
        self.system = system

    def respond(self, message: str, history: Sequence[dict] = ()) -> Iterator[str]:
        """Yield reply chunks. With tools: one final chunk. Without: streamed."""
        msgs: list[Message] = [
            {"role": h["role"], "content": str(h["content"])}
            for h in history
            if h.get("role") in ("user", "assistant")
        ]
        msgs.append({"role": "user", "content": message})

        if self.tools:
            yield self.backend.chat(msgs, self.system, self.tools)
        else:
            yield from self.backend.stream(msgs, self.system)