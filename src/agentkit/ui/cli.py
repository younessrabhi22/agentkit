"""Terminal chat on top of the agent."""
from __future__ import annotations

from agentkit.agent import Agent


def run() -> None:
    print("AgentKit CLI Chat (type 'exit' to quit)\n")

    try:
        agent = Agent()
        messages: list[dict] = []  # chat history, resent on every turn

        while True:
            user_input = input("[You]: ")

            if user_input.strip().lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            print("[AI]: ", end="")
            full_response = ""
            try:
                for chunk in agent.respond(user_input, messages):
                    print(chunk, end="", flush=True)
                    full_response += chunk
            except Exception as exc:  # noqa: BLE001
                print(f"\n⚠️  Temporary problem ({type(exc).__name__}). Please send your message again.\n")
                continue  # keep the session alive, don't touch the history
            print("\n")

            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": full_response})

    except (KeyboardInterrupt, EOFError):  # Ctrl+C or Ctrl+D
        print("\nGoodbye!")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Error: {exc}")  # e.g. missing API key at startup


if __name__ == "__main__":
    run()
