from agentkit.config import Settings
from agentkit.llm.base import LLMBackend, Message

class FallbackBackend(LLMBackend):
    """Try the primary backend; if it fails, use the secondary one."""
    name = "fallback"

    def __init__(self, primary: LLMBackend, secondary: LLMBackend):
        self.primary, self.secondary = primary, secondary
        self.name = f"{primary.name}+{secondary.name}"

    def chat(self, messages, system="", tools=None) -> str:
        try:
            return self.primary.chat(messages, system, tools)
        except Exception:  # noqa: BLE001
            return self.secondary.chat(messages, system)

    def stream(self, messages, system=""):
        started = False
        try:
            for chunk in self.primary.stream(messages, system):
                started = True
                yield chunk
            return
        except Exception:  # noqa: BLE001
            if started:
                raise
        yield from self.secondary.stream(messages, system)

def get_backend(settings: Settings) -> LLMBackend:
    """Pick the engine from LLM_BACKEND. Imports are lazy on purpose."""
    if settings.backend == "gemini":
        from agentkit.llm.gemini_backend import GeminiBackend
        return GeminiBackend(settings)
    if settings.backend == "ollama":
        from agentkit.llm.ollama_backend import OllamaBackend
        return OllamaBackend(settings)
    raise ValueError(f"Unknown LLM_BACKEND={settings.backend!r} (use 'ollama' or 'gemini')")


__all__ = ["get_backend", "LLMBackend", "Message"]
