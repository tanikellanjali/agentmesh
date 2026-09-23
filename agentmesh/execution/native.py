from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from agentmesh.agents.factory import build_agent
from agentmesh.core.graph import build_levels
from agentmesh.execution.base import agent_metrics, assemble
from agentmesh.schemas.agent_spec import AgentSpec

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelBroker
    from agentmesh.core.orchestrator import RunResult
    from agentmesh.core.run_context import RunContext


class NativeEngine:
    """Level-synchronous execution with no third-party dependencies.

    Agents in a dependency level run concurrently; the next level waits for the
    whole previous one. Simple, predictable, and enough for most meshes.
    """

    name = "native"

    def run(
        self,
        message: str,
        agents: list[AgentSpec],
        broker: "ModelBroker | None" = None,
        run_context: "RunContext | None" = None,
        max_workers: int = 4,
        **options: Any,
    ) -> "RunResult":
        from agentmesh.core.orchestrator import AgentExecution

        levels = build_levels(agents)
        context: dict[str, Any] = {}
        executions: list[AgentExecution] = []

        for index, level in enumerate(levels):
            snapshot = dict(context)
            outputs = self._run_level(message, level, snapshot, broker, run_context, max_workers)
            for spec, output in zip(level, outputs):
                context[spec.id] = output
                executions.append(
                    AgentExecution(
                        agent_id=spec.id, output=output, level=index,
                        **agent_metrics(broker, spec.id),
                    )
                )

        return assemble(agents, executions, broker)

    @staticmethod
    def _run_level(message, level, snapshot, broker, run_context, max_workers):
        def execute(spec: AgentSpec) -> dict[str, Any]:
            return build_agent(spec, broker, run_context).run(message, dict(snapshot))

        if len(level) == 1 or max_workers <= 1:
            return [execute(spec) for spec in level]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(level))) as pool:
            return list(pool.map(execute, level))
