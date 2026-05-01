from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.core.capability_matcher import CapabilityMatch, match_capabilities
from agentmesh.core.need_resolver import NeedResolution, resolve_needs
from agentmesh.core.orchestrator import RunResult, run_agents
from agentmesh.core.project_loader import DEFAULT_SPEC_PACKS_DIR, LoadedProject, load_project


@dataclass(frozen=True)
class RuntimeResult:
    project: LoadedProject
    resolution: NeedResolution
    match: CapabilityMatch
    run: RunResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": {
                "id": self.project.spec.id,
                "name": self.project.spec.name,
            },
            "required_capabilities": self.resolution.required_capabilities,
            "reasoning": self.resolution.reasoning,
            "selected_agents": self.run.selected_agents,
            "missing_capabilities": self.match.missing_capabilities,
            "agent_outputs": [
                {
                    "agent_id": execution.agent_id,
                    "output": execution.output,
                }
                for execution in self.run.executions
            ],
            "final_response": self.run.final_response,
        }


def run_project(
    project_id: str,
    message: str,
    spec_packs_dir=DEFAULT_SPEC_PACKS_DIR,
) -> RuntimeResult:
    loaded = load_project(project_id, spec_packs_dir)
    resolution = resolve_needs(message)
    registry = AgentRegistry(loaded.agents)
    match = match_capabilities(resolution.required_capabilities, registry)
    run = run_agents(message, match.selected_agents)

    return RuntimeResult(
        project=loaded,
        resolution=resolution,
        match=match,
        run=run,
    )
