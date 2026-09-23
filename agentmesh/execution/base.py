from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agentmesh.core.graph import build_levels, find_composer
from agentmesh.providers.base import Usage
from agentmesh.schemas.agent_spec import AgentSpec

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelBroker
    from agentmesh.core.orchestrator import AgentExecution, RunResult
    from agentmesh.core.run_context import RunContext


@runtime_checkable
class ExecutionEngine(Protocol):
    """How a resolved set of agents actually gets run.

    AgentMesh owns *which* agents run and *why*; the engine owns *how*. Keeping
    that seam means adopting LangGraph is a configuration choice rather than a
    rewrite - and the zero-dependency path still works for anyone who wants it.
    """

    name: str

    def run(
        self,
        message: str,
        agents: list[AgentSpec],
        broker: "ModelBroker | None" = None,
        run_context: "RunContext | None" = None,
        **options: Any,
    ) -> "RunResult": ...


def agent_metrics(broker: "ModelBroker | None", agent_id: str) -> dict[str, Any]:
    recorder = getattr(broker, "recorder", None)
    if recorder is None:
        return {"usage": Usage(), "cost": 0.0, "llm_calls": 0, "retries": 0}
    return {
        "usage": recorder.usage_for(agent_id),
        "cost": recorder.cost_for(agent_id),
        "llm_calls": len(recorder.for_agent(agent_id)),
        "retries": recorder.retries_for(agent_id),
    }


def assemble(
    agents: list[AgentSpec],
    executions: list["AgentExecution"],
    broker: "ModelBroker | None",
) -> "RunResult":
    """Shared result assembly so every engine reports identically."""
    from agentmesh.core.orchestrator import RunResult

    levels = build_levels(agents)
    composer = find_composer(agents)
    final_agent_id = composer.id if composer else (executions[-1].agent_id if executions else None)
    final_response = next(
        (e.output for e in executions if e.agent_id == final_agent_id), {}
    )

    recorder = getattr(broker, "recorder", None)
    return RunResult(
        selected_agents=[e.agent_id for e in executions],
        executions=executions,
        final_response=final_response,
        levels=[[a.id for a in level] for level in levels],
        usage=recorder.usage if recorder else Usage(),
        cost=recorder.cost if recorder else 0.0,
        llm_calls=len(recorder.calls) if recorder else 0,
        retries=recorder.retries if recorder else 0,
        final_agent_id=final_agent_id,
    )
