from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentmesh import __version__
from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.core.help_catalog import list_help_entries
from agentmesh.core.model_settings import list_project_models, write_env_file
from agentmesh.core.project_generator import draft_project, write_project_files
from agentmesh.core.project_loader import DEFAULT_SPEC_PACKS_DIR, list_projects, load_project
from agentmesh.core.runtime import run_project
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


@app.command("help-functions")
def help_functions_command() -> None:
    table = Table(title="AgentMesh Functions")
    table.add_column("Function")
    table.add_column("What it does")
    table.add_column("Example")

    for entry in list_help_entries():
        table.add_row(entry.command, entry.description, entry.example)

    console.print(table)


@app.command("configure-models")
def configure_models_command(
    env_path: Path = typer.Option(Path(".env"), "--env-path"),
) -> None:
    values = {
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "LOCAL_MODEL_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "DEFAULT_LOCAL_MODEL": "gemma3:4b",
        "AGENTMESH_RUN_DIR": ".runs",
    }

    console.print("[bold]Model Provider Setup[/bold]")
    if typer.confirm("Do you have access to OpenAI models?", default=False):
        values["OPENAI_API_KEY"] = typer.prompt("OpenAI API key", hide_input=True)
    if typer.confirm("Do you have access to Anthropic models?", default=False):
        values["ANTHROPIC_API_KEY"] = typer.prompt("Anthropic API key", hide_input=True)
    if typer.confirm("Do you have access to Google Gemini models?", default=False):
        values["GOOGLE_API_KEY"] = typer.prompt("Google API key", hide_input=True)
    if typer.confirm("Do you want to use a local Ollama model?", default=True):
        values["LOCAL_MODEL_PROVIDER"] = "ollama"
        values["OLLAMA_BASE_URL"] = typer.prompt(
            "Ollama base URL",
            default=values["OLLAMA_BASE_URL"],
        )
        values["DEFAULT_LOCAL_MODEL"] = typer.prompt(
            "Default local model",
            default=values["DEFAULT_LOCAL_MODEL"],
        )

    write_env_file(values, env_path)
    console.print(f"[green]Wrote model settings to {env_path}[/green]")


@app.command("list-models")
def list_models_command(
    project: str = typer.Option("data_ops", "--project", "-p"),
    spec_packs_dir: Path = typer.Option(DEFAULT_SPEC_PACKS_DIR, "--spec-packs-dir"),
) -> None:
    try:
        models = list_project_models(project, spec_packs_dir)
    except SpecLoadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table(title=f"Models: {project}")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Available")
    table.add_column("Input $/1K")
    table.add_column("Output $/1K")

    for model in models:
        table.add_row(
            str(model["model_id"]),
            str(model["provider"]),
            "yes" if model["available"] else "no",
            str(model["cost_per_1k_input_tokens"]),
            str(model["cost_per_1k_output_tokens"]),
        )

    console.print(table)


@app.command("init-project")
def init_project_command(
    description: str = typer.Argument(..., help="Description of the agent mesh to build."),
    project_id: str | None = typer.Option(None, "--project-id"),
    spec_packs_dir: Path = typer.Option(DEFAULT_SPEC_PACKS_DIR, "--spec-packs-dir"),
    agents_dir: Path = typer.Option(Path("agentmesh/agents"), "--agents-dir"),
    create_venv: bool = typer.Option(False, "--create-venv", help="Create a .venv inside the generated project directory."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Write the generated files without prompting."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing generated files."),
) -> None:
    project = draft_project(description, project_id)

    console.print(f"[bold]Draft Project[/bold]: {project.name}")
    console.print(project.description)

    table = Table(title="Draft Agents")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Capabilities")
    table.add_column("Tools")

    for agent in project.agents:
        table.add_row(
            agent.id,
            agent.name,
            ", ".join(agent.capabilities),
            ", ".join(agent.tools),
        )

    console.print(table)

    if not yes and not typer.confirm("Generate this project?", default=True):
        console.print("[yellow]No files written.[/yellow]")
        raise typer.Exit()

    try:
        written = write_project_files(
            project,
            spec_packs_dir=spec_packs_dir,
            agents_dir=agents_dir,
            overwrite=overwrite,
            create_venv=create_venv,
        )
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[green]Generated {project.id}[/green]")
    for path in written:
        console.print(f"- {path}")


@app.command("run")
def run_command(
    message: str = typer.Argument(..., help="User request to route through the agent mesh."),
    project: str = typer.Option(..., "--project", "-p"),
    spec_packs_dir: Path = typer.Option(DEFAULT_SPEC_PACKS_DIR, "--spec-packs-dir"),
) -> None:
    try:
        result = run_project(project, message, spec_packs_dir)
    except SpecLoadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[bold]Project[/bold]: {result.project.spec.name}")
    console.print(f"[bold]Request[/bold]: {message}")
    console.print(f"[bold]Reasoning[/bold]: {result.resolution.reasoning}")

    table = Table(title="Selected Agents")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Capabilities")

    agents_by_id = {agent.id: agent for agent in result.match.selected_agents}
    for agent_id in result.run.selected_agents:
        agent = agents_by_id[agent_id]
        table.add_row(agent.id, agent.name, ", ".join(agent.capabilities))

    console.print(table)

    if result.match.missing_capabilities:
        console.print(
            "[yellow]Missing capabilities:[/yellow] "
            + ", ".join(result.match.missing_capabilities)
        )

    console.print("[bold]Agent Outputs[/bold]")
    for execution in result.run.executions:
        console.print(f"[cyan]{execution.agent_id}[/cyan]")
        console.print_json(data=execution.output)


if __name__ == "__main__":
    app()
