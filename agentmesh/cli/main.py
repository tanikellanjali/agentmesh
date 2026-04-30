from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentmesh import __version__
from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.core.project_loader import DEFAULT_SPEC_PACKS_DIR, list_projects, load_project
from agentmesh.core.spec_loader import SpecLoadError

app = typer.Typer(help="Spec-driven agent mesh runtime.")
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"agentmesh {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        help="Show AgentMesh version.",
    ),
) -> None:
    pass


@app.command("list-projects")
def list_projects_command(
    spec_packs_dir: Path = typer.Option(DEFAULT_SPEC_PACKS_DIR, "--spec-packs-dir"),
) -> None:
    projects = list_projects(spec_packs_dir)
    table = Table(title="AgentMesh Projects")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Entrypoint")

    for project in projects:
        table.add_row(project.id, project.name, project.entrypoint)

    console.print(table)


@app.command("list-agents")
def list_agents_command(
    project: str = typer.Option(..., "--project", "-p"),
    include_disabled: bool = typer.Option(False, "--include-disabled"),
    spec_packs_dir: Path = typer.Option(DEFAULT_SPEC_PACKS_DIR, "--spec-packs-dir"),
) -> None:
    try:
        loaded = load_project(project, spec_packs_dir)
    except SpecLoadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    registry = AgentRegistry(loaded.agents)
    table = Table(title=f"Agents: {loaded.spec.name}")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Capabilities")
    table.add_column("Enabled")

    for agent in registry.list(include_disabled=include_disabled):
        table.add_row(
            agent.id,
            agent.name,
            agent.category,
            ", ".join(agent.capabilities),
            "yes" if agent.enabled else "no",
        )

    console.print(table)


@app.command("validate-spec")
def validate_spec_command(
    project: str = typer.Option(..., "--project", "-p"),
    spec_packs_dir: Path = typer.Option(DEFAULT_SPEC_PACKS_DIR, "--spec-packs-dir"),
) -> None:
    try:
        loaded = load_project(project, spec_packs_dir)
    except SpecLoadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(
        f"[green]OK[/green] {loaded.spec.id}: "
        f"{len(loaded.agents)} agent specs loaded from {loaded.root}"
    )


if __name__ == "__main__":
    app()
