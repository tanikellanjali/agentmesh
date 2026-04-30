from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from agentmesh.schemas.agent_spec import AgentSpec

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class SpecLoadError(RuntimeError):
    """Raised when a YAML spec cannot be loaded or validated."""


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecLoadError(f"Spec file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SpecLoadError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SpecLoadError(f"Spec file must contain a YAML mapping: {path}")
    return raw


def load_typed_yaml(path: Path, schema: type[SchemaT]) -> SchemaT:
    try:
        return schema.model_validate(read_yaml(path))
    except ValidationError as exc:
        raise SpecLoadError(f"Spec validation failed for {path}: {exc}") from exc


def load_agent_spec(path: Path) -> AgentSpec:
    return load_typed_yaml(path, AgentSpec)


def load_agent_specs(agent_dir: Path) -> list[AgentSpec]:
    if not agent_dir.exists():
        raise SpecLoadError(f"Agent directory not found: {agent_dir}")

    specs = [load_agent_spec(path) for path in sorted(agent_dir.glob("*.yaml"))]
    ids = [spec.id for spec in specs]
    duplicates = sorted({agent_id for agent_id in ids if ids.count(agent_id) > 1})
    if duplicates:
        raise SpecLoadError(f"Duplicate agent ids in {agent_dir}: {', '.join(duplicates)}")
    return specs
