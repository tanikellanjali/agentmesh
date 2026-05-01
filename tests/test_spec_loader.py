from pathlib import Path

import pytest

from agentmesh.core.project_loader import list_projects, load_project
from agentmesh.core.spec_loader import SpecLoadError, load_agent_spec


ROOT = Path(__file__).resolve().parents[1]
SPEC_PACKS = ROOT / "spec_packs"


def test_list_projects_loads_data_ops() -> None:
    projects = list_projects(SPEC_PACKS)

    assert [project.id for project in projects] == ["data_ops"]
    assert all(project.agent_dir == "agents" for project in projects)


def test_load_project_loads_agent_specs() -> None:
    loaded = load_project("data_ops", SPEC_PACKS)

    assert loaded.spec.id == "data_ops"
    assert len(loaded.agents) == 7
    assert {agent.id for agent in loaded.agents} >= {
        "compliance_agent",
        "sql_analyzer_agent",
        "troubleshooting_agent",
    }


def test_missing_project_raises_clear_error() -> None:
    with pytest.raises(SpecLoadError, match="Project not found"):
        load_project("missing", SPEC_PACKS)


def test_agent_spec_validation_rejects_unknown_fields(tmp_path: Path) -> None:
    spec = tmp_path / "agent.yaml"
    spec.write_text(
        """
id: invalid_agent
name: Invalid Agent
category: test
description: Invalid test fixture.
capabilities: []
input_contract: {}
output_contract: {}
models: {}
surprise: nope
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(SpecLoadError, match="Spec validation failed"):
        load_agent_spec(spec)
