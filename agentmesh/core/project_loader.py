from pathlib import Path

from agentmesh.core.spec_loader import SpecLoadError, load_agent_specs, load_typed_yaml
from agentmesh.schemas.agent_spec import AgentSpec
from agentmesh.schemas.project_spec import ProjectSpec


DEFAULT_SPEC_PACKS_DIR = Path("spec_packs")


class LoadedProject:
    def __init__(self, root: Path, spec: ProjectSpec, agents: list[AgentSpec]) -> None:
        self.root = root
        self.spec = spec
        self.agents = agents


def list_projects(spec_packs_dir: Path = DEFAULT_SPEC_PACKS_DIR) -> list[ProjectSpec]:
    if not spec_packs_dir.exists():
        return []

    projects: list[ProjectSpec] = []
    for project_yaml in sorted(spec_packs_dir.glob("*/project.yaml")):
        projects.append(load_typed_yaml(project_yaml, ProjectSpec))
    return projects


def load_project(project_id: str, spec_packs_dir: Path = DEFAULT_SPEC_PACKS_DIR) -> LoadedProject:
    project_root = spec_packs_dir / project_id
    project_yaml = project_root / "project.yaml"
    if not project_yaml.exists():
        raise SpecLoadError(f"Project not found: {project_id}")

    project = load_typed_yaml(project_yaml, ProjectSpec)
    agent_dir = project_root / project.agent_dir
    agents = load_agent_specs(agent_dir)
    return LoadedProject(root=project_root, spec=project, agents=agents)
