import typer

from semlink.core import run_task
from semlink.errors import AppError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Automatic semantic note linking for graph visualization and quick topic access",
)


@app.command()
def run(
    name: str = typer.Argument(..., help="Task name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not execute"),
) -> None:
    """
    Run a task.
    """
    try:
        result = run_task(name=name, dry_run=dry_run)
    except AppError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(result)
