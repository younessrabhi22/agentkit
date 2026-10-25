"""Entry point: `uv run agentkit [chat|ui|check]`."""
from __future__ import annotations

import argparse


def check() -> int:
    """Health test: config loads, backend is reachable, model answers."""
    from agentkit.config import get_settings
    from agentkit.llm import get_backend

    s = get_settings()
    model = s.ollama_model if s.backend == "ollama" else s.gemini_model
    print(f"backend={s.backend} model={model}")
    try:
        backend = get_backend(s)
        reply = backend.chat([{"role": "user", "content": "Reply with exactly one word: ready"}])
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")
        return 1
    print(f"reply: {reply}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="agentkit")
    parser.add_argument("command", nargs="?", default="chat", choices=["chat", "ui", "check"])
    args = parser.parse_args()

    if args.command == "check":
        return check()
    if args.command == "ui":
        from agentkit.ui.gradio_app import launch
        launch()
        return 0

    from agentkit.ui.cli import run  # default: "chat"
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())