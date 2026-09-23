from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentmesh.agents.factory import build_agent
from agentmesh.core.graph import build_levels, find_composer
from agentmesh.providers.base import Usage
from agentmesh.schemas.agent_spec import AgentSpec

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelBroker


@dataclass(frozen=True)
class AgentExecution:
    agent_id: str
    output: dict[str, Any]
    level: int = 0
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    llm_calls: int = 0
    retries: int = 0


@dataclass(frozen=True)
class RunResult:
    selected_agents: list[str]
    executions: list[AgentExecution]
    final_response: dict[str, Any]
    levels: list[list[str]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    llm_calls: int = 0
    retries: int = 0
    final_agent_id: str | None = None


def order_agents(agents: list[AgentSpec]) -> list[AgentSpec]:
    """Flatten the dependency graph into execution order."""
    return [agent for level in build_levels(agents) for agent in level]


def run_agents(
    message: str,
    agents: list[AgentSpec],
    broker: "ModelBroker | None" = None,
    max_workers: int = 4,
) -> RunResult:
    levels = build_levels(agents)
    context: dict[str, Any] = {}
    executions: list[AgentExecution] = []

    for index, level in enumerate(levels):
        # Agents in a level are independent: each reads a snapshot of what
        # earlier levels produced, and their outputs merge once the level ends.
        snapshot = dict(context)
        outputs = _run_level(message, level, snapshot, broker, max_workers)

        for spec, output in zip(level, outputs):
            context[spec.id] = output
            executions.append(
                AgentExecution(
                    agent_id=spec.id,
                    output=output,
                    level=index,
                    **_metrics(broker, spec.id),
                )
            )

    composer = find_composer(agents)
    final_agent_id = composer.id if composer else (executions[-1].agent_id if executions else None)
    final_response = next(
        (execution.output for execution in executions if execution.agent_id == final_agent_id),
        {},
    )

    recorder = getattr(broker, "recorder", None)
    return RunResult(
        selected_agents=[execution.agent_id for execution in executions],
        executions=executions,
        final_response=final_response,
        levels=[[agent.id for agent in level] for level in levels],
        usage=recorder.usage if recorder else Usage(),
        cost=recorder.cost if recorder else 0.0,
        llm_calls=len(recorder.calls) if recorder else 0,
        retries=recorder.retries if recorder else 0,
        final_agent_id=final_agent_id,
    )


def _run_level(
    message: str,
    level: list[AgentSpec],
    snapshot: dict[str, Any],
    broker: "ModelBroker | None",
    max_workers: int,
) -> list[dict[str, Any]]:
    def execute(spec: AgentSpec) -> dict[str, Any]:
        return build_agent(spec, broker).run(message, dict(snapshot))

    if len(level) == 1 or max_workers <= 1:
        return [execute(spec) for spec in level]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(level))) as pool:
        return list(pool.map(execute, level))


def _metrics(broker: "ModelBroker | None", agent_id: str) -> dict[str, Any]:
    recorder = getattr(broker, "recorder", None)
    if recorder is None:
        return {"usage": Usage(), "cost": 0.0, "llm_calls": 0, "retries": 0}
    return {
        "usage": recorder.usage_for(agent_id),
        "cost": recorder.cost_for(agent_id),
        "llm_calls": len(recorder.for_agent(agent_id)),
        "retries": recorder.retries_for(agent_id),
    }
