from __future__ import annotations

from pathlib import Path

import yaml

from agentmesh.schemas.agent_spec import AgentSpec


def persist_agents(
    agents: list[AgentSpec],
    project_root: Path,
    agent_dir: str = "agents",
    overwrite: bool = False,
) -> list[Path]:
    """Write agent specs into a project's agent directory.

    Synthesized agents are ephemeral by default; this is only called when the
    caller explicitly asks for them to be stored.
    """
    target_dir = project_root / agent_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for agent in agents:
        path = target_dir / f"{agent.id}.yaml"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Agent spec already exists: {path}")
        path.write_text(
            yaml.safe_dump(agent.model_dump(), sort_keys=False),
            encoding="utf-8",
        )
        written.append(path)
    return written
