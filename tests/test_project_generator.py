from agentmesh.core.project_generator import draft_project, write_project_files
from agentmesh.core.project_loader import load_project
from agentmesh.core.runtime import run_project


def test_draft_project_creates_confirmable_agent_plan() -> None:
    project = draft_project("Build a customer support triage mesh", "support_triage")

    assert project.id == "support_triage"
    assert [agent.id for agent in project.agents] == [
        "intake_agent",
        "research_agent",
        "planning_agent",
        "report_agent",
        "troubleshooting_agent",
    ]
    assert project.agents[0].tools == ["input_parser", "constraint_extractor"]


def test_write_project_files_generates_valid_specs_and_executor(tmp_path) -> None:
    project = draft_project("Build a customer support triage mesh", "support_triage")
    spec_packs_dir = tmp_path / "spec_packs"
    agents_dir = tmp_path / "agentmesh" / "agents"

    written = write_project_files(project, spec_packs_dir, agents_dir)

    loaded = load_project("support_triage", spec_packs_dir)

    assert loaded.spec.id == "support_triage"
    assert len(loaded.agents) == 5
    assert agents_dir / "support_triage_agents.py" in written
    assert spec_packs_dir / "support_triage" / "README.md" in written
    assert spec_packs_dir / "support_triage" / "requirements.txt" in written
    assert spec_packs_dir / "support_triage" / ".gitignore" in written
    assert spec_packs_dir / "support_triage" / ".env.example" in written


def test_generated_project_can_run_with_generic_executor(tmp_path) -> None:
    project = draft_project("Build a customer support triage mesh", "support_triage")
    spec_packs_dir = tmp_path / "spec_packs"
    agents_dir = tmp_path / "agentmesh" / "agents"
    write_project_files(project, spec_packs_dir, agents_dir)

    result = run_project(
        "support_triage",
        "Create a triage workflow and recommendation for a refund request.",
        spec_packs_dir,
        provider="mock",
    )

    assert result.run.selected_agents == [
        "intake_agent",
        "planning_agent",
        "report_agent",
    ]
    assert result.run.executions[0].output["agent_id"] == "intake_agent"
