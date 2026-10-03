"""CLI chat loop for the agentic RAG assistant."""
from rich.console import Console
from rich.markdown import Markdown

from config import Config
from agent.agent import AgenticRAGAgent

console = Console()


def main() -> None:
    try:
        Config.validate()
    except ValueError as e:
        console.print(f"[bold red]Config error:[/bold red] {e}")
        return

    console.print(
        f"[bold cyan]Agentic RAG over Live Data[/bold cyan] "
        f"(model: {Config.GROQ_MODEL})"
    )
    console.print("Type 'exit' to quit, 'reset' to clear conversation memory.\n")

    agent = AgenticRAGAgent()

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            console.print("Bye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            console.print("[dim]Conversation memory cleared.[/dim]\n")
            continue

        with console.status("[dim]thinking...[/dim]", spinner="dots"):
            answer = agent.ask(user_input, verbose=True)

        console.print("[bold magenta]Agent:[/bold magenta]")
        console.print(Markdown(answer))
        console.print()


if __name__ == "__main__":
    main()
