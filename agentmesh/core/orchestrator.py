from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentmesh.core.graph import build_levels
from agentmesh.providers.base import Usage
from agentmesh.schemas.agent_spec import AgentSpec

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelBroker
    from agentmesh.core.run_context import RunContext


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
    broker: ModelBroker | None = None,
    max_workers: int = 4,
    run_context: RunContext | None = None,
    engine: Any = None,
) -> RunResult:
    """Run a resolved mesh. Defaults to the dependency-free native engine."""
    from agentmesh.execution.native import NativeEngine

    return (engine or NativeEngine()).run(
        message,
        agents,
        broker=broker,
        run_context=run_context,
        max_workers=max_workers,
    )


def _metrics(broker: ModelBroker | None, agent_id: str) -> dict[str, Any]:
    from agentmesh.execution.base import agent_metrics

    return agent_metrics(broker, agent_id)
