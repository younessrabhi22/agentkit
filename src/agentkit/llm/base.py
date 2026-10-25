"""The one interface every LLM backend must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Literal, Sequence, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class LLMBackend(ABC):
    """Swap engines without touching the rest of the app."""

    name: str

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        system: str = "",
        tools: Sequence[Callable] | None = None,
    ) -> str:
        """Return the full answer as one string."""

    @abstractmethod
    def stream(self, messages: Sequence[Message], system: str = "") -> Iterator[str]:
        """Yield the answer in text chunks."""