from __future__ import annotations

from agentmesh.schemas.agent_spec import AgentSpec

COMPOSER_CAPABILITY = "response_composition"


class GraphError(RuntimeError):
    """Raised when agent dependencies cannot form a runnable graph."""


def is_composer(agent: AgentSpec) -> bool:
    return COMPOSER_CAPABILITY in agent.capabilities


def find_composer(agents: list[AgentSpec]) -> AgentSpec | None:
    """The agent that writes the final answer, if the mesh declares one."""
    composers = sorted((a for a in agents if is_composer(a)), key=lambda a: a.id)
    return composers[0] if composers else None


def _depends_transitively_on(
    start: str, target: str, explicit: dict[str, list[str]], seen: set[str] | None = None
) -> bool:
    """Is `target` reachable from `start` through explicit depends_on edges?"""
    seen = seen if seen is not None else set()
    if start in seen:
        return False
    seen.add(start)
    for dep in explicit.get(start, []):
        if dep == target or _depends_transitively_on(dep, target, explicit, seen):
            return True
    return False


def dependencies(agent: AgentSpec, agents: list[AgentSpec]) -> list[str]:
    """Explicit `depends_on` wins; otherwise a composer waits for everyone else.

    A composer's implicit dependencies exclude anything placed downstream of it
    by an explicit `depends_on`, so declaring "run after the report" cannot
    create a cycle against the composer's own implicit edges.
    """
    if agent.depends_on:
        return list(agent.depends_on)
    if not is_composer(agent):
        return []

    explicit = {other.id: list(other.depends_on) for other in agents}
    return [
        other.id
        for other in agents
        if other.id != agent.id
        and not is_composer(other)
        and not _depends_transitively_on(other.id, agent.id, explicit)
    ]


def build_levels(agents: list[AgentSpec]) -> list[list[AgentSpec]]:
    """Group agents into dependency levels via a topological sort.

    Agents in the same level have no dependency on each other and may run
    concurrently. Within a level, order is by id so runs are reproducible.
    """
    by_id = {agent.id: agent for agent in agents}
    pending: dict[str, set[str]] = {}

    for agent in agents:
        deps = set(dependencies(agent, agents))
        unknown = sorted(dep for dep in deps if dep not in by_id)
        if unknown:
            raise GraphError(
                f"Agent '{agent.id}' depends on unknown agent(s): {', '.join(unknown)}"
            )
        if agent.id in deps:
            raise GraphError(f"Agent '{agent.id}' depends on itself")
        pending[agent.id] = deps

    levels: list[list[AgentSpec]] = []
    resolved: set[str] = set()

    while pending:
        ready = sorted(agent_id for agent_id, deps in pending.items() if deps <= resolved)
        if not ready:
            cycle = ", ".join(sorted(pending))
            raise GraphError(f"Circular dependency between agents: {cycle}")

        levels.append([by_id[agent_id] for agent_id in ready])
        resolved.update(ready)
        for agent_id in ready:
            del pending[agent_id]

    return levels
