from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentmesh.agents.factory import build_agent
from agentmesh.schemas.agent_spec import AgentSpec


DEFAULT_AGENT_ORDER = {
    "intake_agent": 5,
    "research_agent": 15,
    "planning_agent": 35,
    "report_agent": 1000,
    "troubleshooting_agent": 1100,
}


@dataclass(frozen=True)
class AgentExecution:
    agent_id: str
    output: dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    selected_agents: list[str]
    executions: list[AgentExecution]
    final_response: dict[str, Any]


def order_agents(agents: list[AgentSpec]) -> list[AgentSpec]:
    return sorted(agents, key=lambda agent: (DEFAULT_AGENT_ORDER.get(agent.id, 100), agent.id))


def run_agents(message: str, agents: list[AgentSpec]) -> RunResult:
    context: dict[str, Any] = {}
    executions: list[AgentExecution] = []

    for spec in order_agents(agents):
        executor = build_agent(spec)
        output = executor.run(message, context)
        executions.append(AgentExecution(agent_id=spec.id, output=output))

    final_response = executions[-1].output if executions else {}
    return RunResult(
        selected_agents=[agent.id for agent in order_agents(agents)],
        executions=executions,
        final_response=final_response,
    )
