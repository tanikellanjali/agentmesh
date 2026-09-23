from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentmesh.core.runtime import RuntimeResult

DEFAULT_RUN_DIR = Path(os.environ.get("AGENTMESH_RUN_DIR", ".runs"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows), encoding="utf-8"
    )


def persist_run(
    result: RuntimeResult,
    message: str,
    run_dir: Path | str = DEFAULT_RUN_DIR,
) -> Path:
    """Write a run to disk as inspectable artefacts.

    One directory per run, split so each file is useful on its own: `run.json`
    for the shape of the run, the jsonl files for anything you want to load into
    a table later, and `final_response.json` for the answer itself.
    """
    root = Path(run_dir) / result.recorder.run_id
    root.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()

    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": result.recorder.run_id,
                "started_at": datetime.now(UTC).isoformat(),
                "project": payload["project"],
                "principal": payload.get("principal"),
                "request": message,
                "required_capabilities": payload["required_capabilities"],
                "reasoning": payload["reasoning"],
                "selected_agents": payload["selected_agents"],
                "missing_capabilities": payload["missing_capabilities"],
                "synthesized_agents": payload["synthesized_agents"],
                "graph": payload["graph"],
                "usage": payload["usage"],
                "data_cost_usd": payload.get("data_cost_usd", 0.0),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    _write_jsonl(root / "llm_calls.jsonl", [c.as_dict() for c in result.recorder.calls])
    _write_jsonl(root / "tool_runs.jsonl", [t.as_dict() for t in result.recorder.tool_runs])
    _write_jsonl(root / "data_ops.jsonl", [d.as_dict() for d in result.recorder.data_ops])
    _write_jsonl(
        root / "agent_outputs.jsonl",
        [
            {"agent_id": e.agent_id, "level": e.level, "output": e.output}
            for e in result.run.executions
        ],
    )
    (root / "final_response.json").write_text(
        json.dumps(result.run.final_response, indent=2, default=str), encoding="utf-8"
    )
    return root


def list_runs(run_dir: Path | str = DEFAULT_RUN_DIR, limit: int = 20) -> list[dict[str, Any]]:
    root = Path(run_dir)
    if not root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
        if len(runs) >= limit:
            break
    return runs


def load_run(run_id: str, run_dir: Path | str = DEFAULT_RUN_DIR) -> dict[str, Any]:
    root = Path(run_dir) / run_id
    if not (root / "run.json").exists():
        raise FileNotFoundError(f"no run recorded under {root}")

    def read_jsonl(name: str) -> list[dict[str, Any]]:
        path = root / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    run["llm_calls"] = read_jsonl("llm_calls.jsonl")
    run["tool_runs"] = read_jsonl("tool_runs.jsonl")
    run["data_ops"] = read_jsonl("data_ops.jsonl")
    run["agent_outputs"] = read_jsonl("agent_outputs.jsonl")
    final = root / "final_response.json"
    run["final_response"] = json.loads(final.read_text(encoding="utf-8")) if final.exists() else {}
    return run
