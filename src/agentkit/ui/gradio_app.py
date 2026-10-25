"""Web UI. Streaming chat on top of the agent."""
from __future__ import annotations

import gradio as gr

from agentkit.agent import Agent
from agentkit.config import get_settings

WARNING = "⚠️ The model had a temporary problem. Please send your message again."


def _clean(history: list[dict]) -> list[dict]:
    """Remove failed turns (the user message + the warning reply) before sending history to the model."""
    out: list[dict] = []
    for h in history:
        content = str(h.get("content", ""))
        if h.get("role") == "assistant" and content.endswith(WARNING):
            if out and out[-1].get("role") == "user":
                out.pop()  # drop the user message that got no real answer
            continue
        out.append(h)
    return out


def build_app(agent: Agent) -> gr.ChatInterface:
    def fn(message: str, history: list[dict]):
        partial = ""
        try:
            for chunk in agent.respond(message, _clean(history)):
                partial += chunk
                yield partial  # Gradio expects the growing text, not just the new chunk
        except Exception:  # noqa: BLE001
            # A server hiccup must never break the chat: show a friendly line instead
            yield f"{partial}\n\n{WARNING}" if partial else WARNING

    return gr.ChatInterface(
        fn,
        type="messages",
        title="AgentKit",
        description=f"Backend: {agent.backend.name}",
    )


def launch() -> None:
    s = get_settings()
    build_app(Agent()).launch(server_name=s.host, server_port=s.port)
