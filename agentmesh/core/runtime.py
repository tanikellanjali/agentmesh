from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentmesh.access.authorizer import AllowAll, Authorizer
from agentmesh.access.principal import ANONYMOUS, Principal
from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.core.agent_store import persist_agents
from agentmesh.core.capability_matcher import CapabilityMatch, match_capabilities
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.model_settings import configured_env
from agentmesh.core.run_context import RunContext
from agentmesh.core.telemetry import RunRecorder
from agentmesh.core.need_resolver import NeedResolution, resolve_needs
from agentmesh.core.orchestrator import RunResult, run_agents
from agentmesh.core.project_loader import DEFAULT_SPEC_PACKS_DIR, LoadedProject, load_project
from agentmesh.core.spec_loader import read_yaml
from agentmesh.data.context import SourceRegistry
from agentmesh.providers.retry import RetryPolicy


@dataclass(frozen=True)
class RuntimeResult:
    project: LoadedProject
    resolution: NeedResolution
    match: CapabilityMatch
    run: RunResult
    recorder: RunRecorder = field(default_factory=RunRecorder)
    principal: Principal = ANONYMOUS
    stored_agent_paths: list[Path] = field(default_factory=list)

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
            "synthesized_agents": [agent.id for agent in self.match.synthesized_agents],
            "stored_agents": [str(path) for path in self.stored_agent_paths],
            "graph": {
                "levels": self.run.levels,
                "final_agent": self.run.final_agent_id,
            },
            "usage": {
                "input_tokens": self.run.usage.input_tokens,
                "output_tokens": self.run.usage.output_tokens,
                "estimated_cost_usd": round(self.run.cost, 6),
                "llm_calls": self.run.llm_calls,
                "retries": self.run.retries,
            },
            "agent_outputs": [
                {
                    "agent_id": execution.agent_id,
                    "level": execution.level,
                    "output": execution.output,
                    "usage": {
                        "input_tokens": execution.usage.input_tokens,
                        "output_tokens": execution.usage.output_tokens,
                        "estimated_cost_usd": round(execution.cost, 6),
                        "llm_calls": execution.llm_calls,
                        "retries": execution.retries,
                    },
                }
                for execution in self.run.executions
            ],
            "llm_calls": [call.as_dict() for call in self.recorder.calls],
            "run_id": self.recorder.run_id,
            "principal": self.principal.as_dict(),
            "data_ops": [d.as_dict() for d in self.recorder.data_ops],
            "data_cost_usd": round(self.recorder.data_cost, 6),
            "final_response": self.run.final_response,
        }


def build_broker(
    project: LoadedProject,
    provider: str | None = None,
    model: str | None = None,
    credentials: dict[str, str] | None = None,
    recorder: RunRecorder | None = None,
    retry_policy: RetryPolicy | None = None,
) -> ModelBroker:
    routing_path = project.root / project.spec.model_routing
    models = read_yaml(routing_path).get("models", {}) if routing_path.exists() else {}
    env = credentials if credentials is not None else configured_env()

    # Explicit arguments win; otherwise fall back to the user's saved choice.
    return ModelBroker(
        models=models,
        credentials=env,
        provider_override=provider or env.get("AGENTMESH_PROVIDER") or None,
        model_override=model or env.get("AGENTMESH_MODEL") or None,
        recorder=recorder,
        retry_policy=retry_policy,
    )


def run_project(
    project_id: str,
    message: str,
    spec_packs_dir=DEFAULT_SPEC_PACKS_DIR,
    synthesize: bool = True,
    store_agents: bool = False,
    overwrite_stored: bool = False,
    provider: str | None = None,
    model: str | None = None,
    credentials: dict[str, str] | None = None,
    max_cost: float | None = None,
    retry_policy: RetryPolicy | None = None,
    max_workers: int = 4,
    principal: Principal = ANONYMOUS,
    authorizer: Authorizer | None = None,
    sources: SourceRegistry | None = None,
    max_data_cost: float | None = None,
) -> RuntimeResult:
    loaded = load_project(project_id, spec_packs_dir)
    resolution = resolve_needs(message)
    registry = AgentRegistry(loaded.agents)
    match = match_capabilities(
        resolution.required_capabilities,
        registry,
        synthesize=synthesize,
    )
    # No default ceiling: a budget cap exists only when the caller sets one.
    recorder = RunRecorder(max_cost=max_cost)
    broker = build_broker(
        loaded,
        provider=provider,
        model=model,
        credentials=credentials,
        recorder=recorder,
        retry_policy=retry_policy,
    )
    run_context = RunContext(
        principal=principal,
        authorizer=authorizer or AllowAll(),
        sources=sources or SourceRegistry.empty(),
        recorder=recorder,
        max_data_cost=max_data_cost,
    )
    run = run_agents(
        message, match.selected_agents, broker,
        max_workers=max_workers, run_context=run_context,
    )

    stored: list[Path] = []
    if store_agents and match.synthesized_agents:
        stored = persist_agents(
            match.synthesized_agents,
            loaded.root,
            loaded.spec.agent_dir,
            overwrite=overwrite_stored,
        )

    return RuntimeResult(
        project=loaded,
        resolution=resolution,
        match=match,
        run=run,
        recorder=recorder,
        principal=principal,
        stored_agent_paths=stored,
    )
