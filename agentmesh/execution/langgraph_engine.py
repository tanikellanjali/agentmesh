from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from agentmesh.agents.factory import build_agent
from agentmesh.core.graph import build_levels, dependencies
from agentmesh.execution.base import agent_metrics, assemble
from agentmesh.schemas.agent_spec import AgentSpec

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelBroker
    from agentmesh.core.orchestrator import RunResult
    from agentmesh.core.run_context import RunContext


class LangGraphUnavailable(RuntimeError):
    """Raised when the langgraph extra is not installed."""


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer for concurrent nodes writing into shared state."""
    return {**left, **right}


class MeshState(TypedDict):
    message: str
    outputs: Annotated[dict, _merge]


class LangGraphEngine:
    """Executes a mesh as a LangGraph `StateGraph`.

    Two things this buys over the native engine. Edges come straight from each
    agent's `depends_on`, so a node starts the moment *its own* dependencies
    finish rather than waiting for a whole level. And a compiled graph supports
    checkpointing, resume, interrupts and streaming - the parts of durable
    execution that are a distributed-systems problem, not a weekend's work.

    AgentMesh still decides which agents exist and why; LangGraph only runs them.
    """

    name = "langgraph"

    def __init__(self, checkpointer: Any = None, interrupt_before: list[str] | None = None) -> None:
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before

    @staticmethod
    def _require():
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - optional extra
            raise LangGraphUnavailable(
                "LangGraph is not installed. Install with: pip install 'agentmesh[langgraph]'"
            ) from exc
        return StateGraph, START, END

    def build(
        self,
        agents: list[AgentSpec],
        broker: ModelBroker | None = None,
        run_context: RunContext | None = None,
    ):
        StateGraph, START, END = self._require()
        graph = StateGraph(MeshState)
        by_id = {a.id: a for a in agents}

        def make_node(spec: AgentSpec):
            def node(state: MeshState) -> dict[str, Any]:
                upstream = dict(state.get("outputs") or {})
                output = build_agent(spec, broker, run_context).run(state["message"], upstream)
                return {"outputs": {spec.id: output}}

            return node

        for spec in agents:
            graph.add_node(spec.id, make_node(spec))

        has_dependents = set()
        for spec in agents:
            deps = [d for d in dependencies(spec, agents) if d in by_id]
            if deps:
                for dep in deps:
                    graph.add_edge(dep, spec.id)
                    has_dependents.add(dep)
            else:
                graph.add_edge(START, spec.id)

        for spec in agents:
            if spec.id not in has_dependents:
                graph.add_edge(spec.id, END)

        return graph

    def compile(self, agents, broker=None, run_context=None):
        graph = self.build(agents, broker, run_context)
        kwargs: dict[str, Any] = {}
        if self.checkpointer is not None:
            kwargs["checkpointer"] = self.checkpointer
        if self.interrupt_before:
            kwargs["interrupt_before"] = self.interrupt_before
        return graph.compile(**kwargs)

    def run(
        self,
        message: str,
        agents: list[AgentSpec],
        broker: ModelBroker | None = None,
        run_context: RunContext | None = None,
        config: dict[str, Any] | None = None,
        **options: Any,
    ) -> RunResult:
        from agentmesh.core.orchestrator import AgentExecution

        compiled = self.compile(agents, broker, run_context)
        final = compiled.invoke({"message": message, "outputs": {}}, config=config or {})
        outputs = final.get("outputs", {})

        # Report in the same deterministic order the native engine uses.
        levels = build_levels(agents)
        executions = [
            AgentExecution(
                agent_id=spec.id,
                output=outputs.get(spec.id, {}),
                level=index,
                **agent_metrics(broker, spec.id),
            )
            for index, level in enumerate(levels)
            for spec in level
            if spec.id in outputs
        ]
        return assemble(agents, executions, broker)
