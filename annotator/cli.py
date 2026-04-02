"""CLI interface for annotator."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from annotator import __version__
from annotator.config import ExitCode, Settings

TEAL = "#00E5B0"
AMBER = "#c05d3b"

app = typer.Typer(
    name="annotator",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)


def _print_banner(console: Console) -> None:
    """Print the EmbedKombinat branded header."""
    logo = Text()
    logo.append("    /\\  /\\  /\\\n", style="bold")
    logo.append("   /  \\/  \\/  \\\n", style="bold")
    logo.append("  /    \\   \\   \\\n", style="bold")

    title = f"embed kombinat \u00b7 annotator v{__version__}"
    console.print()
    console.print(logo, end="")
    console.print(Panel(title, style=f"bold {TEAL}", width=len(title) + 6))
    console.print()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", help="Pairs per batch claimed from kombinat")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Override auto-selected model (HuggingFace ID)")
    ] = None,
    quantization: Annotated[
        str | None, typer.Option("--quantization", help="Override quantization (awq, fp16, etc.)")
    ] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", help="Override backend (vllm, mlx, llama_cpp)")
    ] = None,
    gpu_memory_utilization: Annotated[
        float, typer.Option("--gpu-memory-utilization", help="GPU memory fraction (vLLM only)")
    ] = 0.9,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Process one pair without submitting")
    ] = False,
) -> None:
    """Start the labeling loop. Default command."""
    if ctx.invoked_subcommand is not None:
        return

    console = Console()
    _print_banner(console)

    from annotator.runner import AnnotatorRunner

    settings = Settings()
    runner = AnnotatorRunner(settings, console)
    exit_code = runner.run(
        batch_size=batch_size if batch_size is not None else settings.batch_size,
        model_override=model,
        quantization_override=quantization,
        backend_override=backend,
        gpu_memory_utilization=gpu_memory_utilization,
        dry_run=dry_run,
    )
    raise typer.Exit(code=exit_code)


@app.command()
def login() -> None:
    """Authenticate with GitHub."""
    from annotator import auth

    console = Console()
    settings = Settings()
    try:
        token = auth.login(settings, console)
        console.print(
            f"  [{TEAL}]\u2713[/{TEAL}] Authenticated as {token.contributor.github_username}"
        )
    except Exception as e:
        console.print(f"  [{AMBER}]\u2717[/{AMBER}] Login failed: {e}")
        raise typer.Exit(code=ExitCode.AUTH_FAILURE) from e


@app.command()
def status() -> None:
    """Show contributor profile and stats."""
    from annotator import auth
    from annotator.client import KombinatClient

    console = Console()
    settings = Settings()
    token = auth.load_token(settings.annotator_home)
    if token is None:
        console.print(f"  [{AMBER}]Not logged in.[/{AMBER}] Run 'annotator login'.")
        raise typer.Exit(code=ExitCode.AUTH_FAILURE)

    client = KombinatClient(token.kombinat_url, token.access_token)
    try:
        profile = client.get_profile()
        console.print(f"  [{TEAL}]\u2713[/{TEAL}] Logged in as {profile.github_username}")
        console.print(f"    Total annotations: {profile.total_annotations}")
        console.print(f"    Reputation score:  {profile.reputation_score:.2f}")
    except Exception as e:
        console.print(f"  [{AMBER}]\u2717[/{AMBER}] Failed to fetch status: {e}")
        raise typer.Exit(code=ExitCode.KOMBINAT_UNREACHABLE) from e
    finally:
        client.close()


@app.command()
def logout() -> None:
    """Remove stored credentials."""
    from annotator import auth

    console = Console()
    settings = Settings()
    auth.delete_token(settings.annotator_home)
    console.print(f"  [{TEAL}]\u2713[/{TEAL}] Logged out.")
