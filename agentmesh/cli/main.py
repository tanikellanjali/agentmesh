from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentmesh import __version__
from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.core.help_catalog import list_help_entries
from agentmesh.core.model_settings import (
    configured_env,
    list_project_models,
    write_env_file,
)
from agentmesh.core.project_generator import draft_project, write_project_files
from agentmesh.core.project_loader import DEFAULT_SPEC_PACKS_DIR, list_projects, load_project
from agentmesh.core.runtime import run_project
from agentmesh.core.spec_loader import SpecLoadError
from agentmesh.core.telemetry import BudgetExceeded, configure_logging
from agentmesh.providers import ProviderError, get_provider, list_providers
from agentmesh.providers.retry import RetryPolicy

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
    values: dict[str, str] = {"AGENTMESH_RUN_DIR": ".runs"}

    console.print("[bold]Model Provider Setup[/bold]")
    console.print("[dim]Only the answers you give are written; other keys are kept.[/dim]")

    if typer.confirm("Do you have an Anthropic (Claude) API key?", default=True):
        values["ANTHROPIC_API_KEY"] = typer.prompt("Anthropic API key", hide_input=True)
    if typer.confirm("Do you have an OpenAI API key?", default=False):
        values["OPENAI_API_KEY"] = typer.prompt("OpenAI API key", hide_input=True)

    provider = typer.prompt(
        "Default provider for every agent ("
        + ", ".join(list_providers())
        + "; blank to use each agent spec's own routing)",
        default="",
        show_default=False,
    ).strip()
    if provider:
        if provider not in list_providers():
            console.print(f"[red]Unknown provider: {provider}[/red]")
            raise typer.Exit(1)
        values["AGENTMESH_PROVIDER"] = provider

    model = typer.prompt(
        "Default routing key or model id (blank to use each agent spec's own default)",
        default="",
        show_default=False,
    ).strip()
    if model:
        values["AGENTMESH_MODEL"] = model

    write_env_file(values, env_path)
    console.print(f"[green]Wrote model settings to {env_path}[/green]")
    console.print("[dim]Keep .env out of version control - it is already gitignored.[/dim]")


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
    table.add_column("Routing key")
    table.add_column("Model id")
    table.add_column("Provider")
    table.add_column("Available")
    table.add_column("Input $/1K")
    table.add_column("Output $/1K")

    for model in models:
        table.add_row(
            str(model["key"]),
            str(model["model_id"]),
            str(model["provider"]),
            "yes" if model["available"] else "no",
            str(model["cost_per_1k_input_tokens"]),
            str(model["cost_per_1k_output_tokens"]),
        )

    console.print(table)


@app.command("check-models")
def check_models_command(
    project: str = typer.Option("data_ops", "--project", "-p"),
    spec_packs_dir: Path = typer.Option(DEFAULT_SPEC_PACKS_DIR, "--spec-packs-dir"),
    live: bool = typer.Option(
        False,
        "--live",
        help="Send one tiny real request per provider to verify the credentials work.",
    ),
) -> None:
    """Check that configured providers can be constructed, and optionally called."""
    try:
        models = list_project_models(project, spec_packs_dir)
    except SpecLoadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    credentials = configured_env()
    table = Table(title=f"Provider check: {project}")
    table.add_column("Routing key")
    table.add_column("Provider")
    table.add_column("Credentials")
    table.add_column("Live call" if live else "")

    failures = 0
    probed: dict[str, str] = {}

    for row in sorted(models, key=lambda item: str(item["key"])):
        provider_name = str(row["provider"])
        try:
            client = get_provider(provider_name, credentials)
            creds = "[green]ok[/green]"
        except ProviderError as exc:
            failures += 1
            table.add_row(str(row["key"]), provider_name, f"[red]{exc}[/red]", "-")
            continue

        result = ""
        if live:
            if provider_name in probed:
                result = probed[provider_name]
            else:
                try:
                    completion = client.complete(
                        system="Reply with the single word: ok",
                        prompt="ok",
                        model=str(row["model_id"]),
                        max_tokens=16,
                    )
                    result = (
                        f"[green]ok[/green] "
                        f"({completion.usage.input_tokens}+{completion.usage.output_tokens} tok)"
                    )
                except ProviderError as exc:
                    failures += 1
                    result = f"[red]{type(exc).__name__}: {exc}[/red]"
                probed[provider_name] = result

        table.add_row(str(row["key"]), provider_name, creds, result)

    console.print(table)
    if live:
        console.print("[dim]Live checks bill your account for a few tokens per provider.[/dim]")
    if failures:
        raise typer.Exit(1)


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
    synthesize: bool = typer.Option(
        True,
        "--synthesize/--no-synthesize",
        help="Spin up ephemeral agents for capabilities no registered agent covers.",
    ),
    save_agents: bool = typer.Option(
        False,
        "--save-agents",
        help="Persist synthesized agents into the project's agent directory.",
    ),
    overwrite_stored: bool = typer.Option(
        False,
        "--overwrite-stored",
        help="Allow --save-agents to replace existing agent specs.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Force every agent onto one provider (e.g. anthropic, openai, mock).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Force every agent onto one routing key or model id.",
    ),
    max_attempts: int = typer.Option(
        3,
        "--max-attempts",
        min=1,
        help="Attempts per model call, including the first (1 disables retries).",
    ),
    max_cost: float | None = typer.Option(
        None,
        "--max-cost",
        help="Stop the run once estimated spend reaches this many USD. No cap by default.",
    ),
    max_workers: int = typer.Option(
        4,
        "--max-workers",
        min=1,
        help="Agents to run concurrently within a dependency level.",
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit one JSON object per model call."),
) -> None:
    configure_logging(level=log_level, json_logs=json_logs)
    try:
        result = run_project(
            project,
            message,
            spec_packs_dir,
            synthesize=synthesize,
            store_agents=save_agents,
            overwrite_stored=overwrite_stored,
            provider=provider,
            model=model,
            max_cost=max_cost,
            retry_policy=RetryPolicy(max_attempts=max_attempts),
            max_workers=max_workers,
        )
    except SpecLoadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except BudgetExceeded as exc:
        console.print(f"[red]Budget stop:[/red] {exc}")
        raise typer.Exit(2) from exc
    except ProviderError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[bold]Project[/bold]: {result.project.spec.name}")
    console.print(f"[bold]Request[/bold]: {message}")
    console.print(f"[bold]Reasoning[/bold]: {result.resolution.reasoning}")

    table = Table(title="Selected Agents")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Capabilities")
    table.add_column("Origin")

    table.add_column("Level")

    synthesized_ids = {agent.id for agent in result.match.synthesized_agents}
    agents_by_id = {agent.id: agent for agent in result.match.selected_agents}
    levels = {agent_id: index for index, ids in enumerate(result.run.levels) for agent_id in ids}
    for agent_id in result.run.selected_agents:
        agent = agents_by_id[agent_id]
        table.add_row(
            agent.id,
            agent.name,
            ", ".join(agent.capabilities),
            "synthesized" if agent_id in synthesized_ids else "spec",
            str(levels.get(agent_id, 0)),
        )

    console.print(table)

    if synthesized_ids and not save_agents:
        console.print(
            "[dim]Synthesized agents are ephemeral. Re-run with --save-agents to keep them.[/dim]"
        )

    if result.stored_agent_paths:
        console.print("[green]Stored agents[/green]")
        for path in result.stored_agent_paths:
            console.print(f"- {path}")

    if result.match.missing_capabilities:
        console.print(
            "[yellow]Missing capabilities:[/yellow] "
            + ", ".join(result.match.missing_capabilities)
        )

    console.print("[bold]Agent Outputs[/bold]")
    for execution in result.run.executions:
        console.print(f"[cyan]{execution.agent_id}[/cyan]")
        console.print_json(data=execution.output)

    cost_table = Table(title=f"Usage - run {result.recorder.run_id}")
    cost_table.add_column("Agent")
    cost_table.add_column("Model")
    cost_table.add_column("Calls", justify="right")
    cost_table.add_column("Retries", justify="right")
    cost_table.add_column("In", justify="right")
    cost_table.add_column("Out", justify="right")
    cost_table.add_column("Cost $", justify="right")

    models_by_agent = {call.agent_id: call.model_key for call in result.recorder.calls}
    for execution in result.run.executions:
        cost_table.add_row(
            execution.agent_id,
            models_by_agent.get(execution.agent_id, "-"),
            str(execution.llm_calls),
            str(execution.retries),
            str(execution.usage.input_tokens),
            str(execution.usage.output_tokens),
            f"{execution.cost:.4f}",
        )

    run_usage = result.run.usage
    cost_table.add_row(
        "[bold]total[/bold]",
        "",
        f"[bold]{result.run.llm_calls}[/bold]",
        f"[bold]{result.run.retries}[/bold]",
        f"[bold]{run_usage.input_tokens}[/bold]",
        f"[bold]{run_usage.output_tokens}[/bold]",
        f"[bold]{result.run.cost:.4f}[/bold]",
    )
    console.print(cost_table)

    if max_cost is not None:
        if result.recorder.over_budget:
            console.print(
                f"[yellow]Budget overshot[/yellow]: spent ${result.run.cost:.4f} "
                f"of ${max_cost:.4f}. Concurrent agents in one level can exceed the "
                f"cap; use --max-workers 1 for a strict ceiling."
            )
        else:
            console.print(f"[dim]Budget: ${result.run.cost:.4f} of ${max_cost:.4f}[/dim]")
    console.print(f"[bold]Final response from[/bold]: {result.run.final_agent_id}")


if __name__ == "__main__":
    app()
