"""Hosted Gemma 4 through the Gemini API (what the MLH Gemma challenge requires)."""
from __future__ import annotations

import time
from typing import Callable, Iterator, Sequence

import httpx
from google import genai
from google.genai import errors, types

from agentkit.config import Settings
from agentkit.llm.base import LLMBackend, Message

RETRY_CODES = {429, 500, 502, 503, 504}


def _retryable(exc: Exception) -> bool:
    """True for temporary problems: network hiccups and Google-side server errors."""
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in RETRY_CODES


class GeminiBackend(LLMBackend):
    name = "gemini"

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is empty. Copy .env.example to .env and add your key."
            )
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.max_attempts = 3  # 1 try + 2 retries, waiting 1s then 2s

    @staticmethod
    def _contents(messages: Sequence[Message], system: str) -> list[types.Content]:
        """Convert our messages to Gemini format.

        The system prompt is prepended to the first user turn instead of using
        `system_instruction`, which is the safest option across Gemma models.
        """
        out, injected = [], not system
        for m in messages:
            text = m["content"]
            if not injected and m["role"] == "user":
                text, injected = f"{system}\n\n{text}", True
            role = "user" if m["role"] == "user" else "model"
            out.append(types.Content(role=role, parts=[types.Part(text=text)]))
        return out

    def chat(self, messages, system="", tools: Sequence[Callable] | None = None) -> str:
        config = types.GenerateContentConfig(tools=list(tools) if tools else None)
        contents = self._contents(messages, system)
        delay = 1.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_attempts or not _retryable(exc):
                    raise
                time.sleep(delay)
                delay *= 2
        return ""  # unreachable, keeps type checkers happy

    def stream(self, messages, system="") -> Iterator[str]:
        contents = self._contents(messages, system)
        delay = 1.0
        for attempt in range(1, self.max_attempts + 1):
            started = False
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model, contents=contents
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as exc:  # noqa: BLE001
                # Never retry after text was shown: it would duplicate the answer
                if started or attempt == self.max_attempts or not _retryable(exc):
                    raise
                time.sleep(delay)
                delay *= 2
