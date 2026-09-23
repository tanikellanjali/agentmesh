from __future__ import annotations

import os
from pathlib import Path

from agentmesh.core.project_loader import DEFAULT_SPEC_PACKS_DIR, load_project
from agentmesh.core.spec_loader import read_yaml

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

SETTING_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "AGENTMESH_PROVIDER",
    "AGENTMESH_MODEL",
    "LOCAL_MODEL_PROVIDER",
    "OLLAMA_BASE_URL",
    "DEFAULT_LOCAL_MODEL",
    "AGENTMESH_RUN_DIR",
)


def load_env_file(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def configured_env(path: Path = Path(".env")) -> dict[str, str]:
    merged = load_env_file(path)
    for key, value in os.environ.items():
        if key.endswith("_API_KEY") or key in SETTING_KEYS:
            merged[key] = value
    return merged


def provider_available(provider: str, env_values: dict[str, str]) -> bool:
    if provider in {"mock", "local"}:
        return True
    if provider == "ollama":
        return bool(env_values.get("OLLAMA_BASE_URL"))
    env_key = PROVIDER_ENV_KEYS.get(provider)
    return bool(env_key and env_values.get(env_key))


def list_project_models(project_id: str, spec_packs_dir=DEFAULT_SPEC_PACKS_DIR) -> list[dict[str, object]]:
    loaded = load_project(project_id, spec_packs_dir)
    model_config_path = loaded.root / loaded.spec.model_routing
    raw_config = read_yaml(model_config_path)
    models = raw_config.get("models", {})
    env_values = configured_env()

    rows: list[dict[str, object]] = []
    for key, config in sorted(models.items()):
        provider = config.get("provider", key.split("/", 1)[0])
        rows.append(
            {
                "key": key,
                "model_id": config.get("model", key.split("/", 1)[-1]),
                "provider": provider,
                "available": provider_available(provider, env_values),
                "cost_per_1k_input_tokens": config.get("cost_per_1k_input_tokens", 0),
                "cost_per_1k_output_tokens": config.get("cost_per_1k_output_tokens", 0),
            }
        )
    return rows


def write_env_file(values: dict[str, str], path: Path = Path(".env")) -> None:
    """Merge settings into `.env`, preserving keys already there.

    Only keys explicitly present in `values` are changed, so re-running setup
    to add one provider never wipes another provider's credentials.
    """
    existing = load_env_file(path)
    merged = {**existing, **{k: v for k, v in values.items() if v is not None}}

    ordered = [key for key in SETTING_KEYS if key in merged]
    extras = [key for key in merged if key not in SETTING_KEYS]
    lines = [f"{key}={merged[key]}" for key in ordered + sorted(extras)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
