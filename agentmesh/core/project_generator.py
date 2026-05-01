from __future__ import annotations

import re
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentmesh.core.project_loader import DEFAULT_SPEC_PACKS_DIR


@dataclass(frozen=True)
class DraftAgent:
    id: str
    name: str
    category: str
    description: str
    capabilities: list[str]
    tools: list[str]


@dataclass(frozen=True)
class DraftProject:
    id: str
    name: str
    description: str
    agents: list[DraftAgent]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "custom_project"


def title_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("_"))


def draft_project(description: str, project_id: str | None = None) -> DraftProject:
    slug = slugify(project_id or description[:48])
    name = f"{title_from_slug(slug)} Agent Pack"
    base_description = description.strip().rstrip(".")

    agents = [
        DraftAgent(
            id="intake_agent",
            name="Intake Agent",
            category=slug,
            description="Extracts the user's request, constraints, goals, and missing details.",
            capabilities=["request_intake", "constraint_extraction"],
            tools=["input_parser", "constraint_extractor"],
        ),
        DraftAgent(
            id="research_agent",
            name="Research Agent",
            category=slug,
            description="Collects domain context and options needed to satisfy the request.",
            capabilities=["domain_research", "option_generation"],
            tools=["knowledge_lookup", "option_ranker"],
        ),
        DraftAgent(
            id="planning_agent",
            name="Planning Agent",
            category=slug,
            description="Turns context and options into an actionable plan.",
            capabilities=["plan_generation", "decision_support"],
            tools=["plan_builder", "tradeoff_analyzer"],
        ),
        DraftAgent(
            id="report_agent",
            name="Report Agent",
            category=slug,
            description="Composes agent outputs into the final user-facing response.",
            capabilities=["report_generation", "response_composition"],
            tools=["markdown_report_writer", "response_contract_validator"],
        ),
        DraftAgent(
            id="troubleshooting_agent",
            name="Troubleshooting Agent",
            category="support",
            description="Explains runtime failures and suggests recovery steps.",
            capabilities=["failure_explanation", "troubleshooting"],
            tools=["failure_classifier", "recovery_suggester"],
        ),
    ]

    return DraftProject(
        id=slug,
        name=name,
        description=f"Generated starter agent mesh for: {base_description}.",
        agents=agents,
    )


def project_spec(project: DraftProject) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "entrypoint": f"{project.id}_request",
        "default_persona": {
            "user": "personas/user_persona.md",
            "use_case": "personas/use_case_persona.md",
        },
        "agent_dir": "agents",
        "model_routing": "models/model_routing.yaml",
        "response_contract": "response_contracts/response.yaml",
    }


def agent_spec(agent: DraftAgent) -> dict[str, Any]:
    return {
        "id": agent.id,
        "name": agent.name,
        "category": agent.category,
        "description": agent.description,
        "capabilities": agent.capabilities,
        "input_contract": {
            "required": ["message"],
            "optional": ["context", "agent_outputs"],
        },
        "output_contract": {
            "format": "json",
            "required_fields": ["summary", "findings", "confidence"],
        },
        "models": {
            "default": "local/gemma-3-4b",
            "fallback": ["openai/gpt-4.1-mini", "google/gemini-flash"],
        },
        "tools": agent.tools,
        "fallback_agents": [],
        "troubleshooting": {
            "enabled": agent.id != "troubleshooting_agent",
            "agent": "troubleshooting_agent",
        },
        "validation": {
            "strict_json": True,
            "confidence_threshold": 0.7,
        },
    }


def executor_source(project: DraftProject) -> str:
    class_prefix = "".join(part.capitalize() for part in project.id.split("_"))
    return f'''from __future__ import annotations

from typing import Any

from agentmesh.agents.base_agent import BaseAgent


class {class_prefix}Agent(BaseAgent):
    """Generated starter executor for the {project.name}."""

    def run(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        output = {{
            "summary": f"{{self.spec.name}} processed the request.",
            "findings": [
                "This is a generated starter agent.",
                "Replace this stub with domain-specific behavior.",
            ],
            "confidence": self._confidence(),
        }}
        context[self.spec.id] = output
        return output
'''


def readme_source(project: DraftProject) -> str:
    return f"""# {project.name}

{project.description}

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate

```bash
agentmesh validate-spec --project {project.id} --spec-packs-dir ..
```

## Run

```bash
agentmesh run --project {project.id} "Describe the user request here" --spec-packs-dir ..
```
"""


def write_project_files(
    project: DraftProject,
    spec_packs_dir: Path = DEFAULT_SPEC_PACKS_DIR,
    agents_dir: Path = Path("agentmesh/agents"),
    overwrite: bool = False,
    create_venv: bool = False,
) -> list[Path]:
    project_root = spec_packs_dir / project.id
    if project_root.exists() and not overwrite:
        raise FileExistsError(f"Project already exists: {project_root}")

    paths = {
        "agents": project_root / "agents",
        "personas": project_root / "personas",
        "models": project_root / "models",
        "response_contracts": project_root / "response_contracts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    files: dict[Path, str] = {
        project_root / "README.md": readme_source(project),
        project_root / ".gitignore": "\n".join(
            [
                "__pycache__/",
                "*.py[cod]",
                ".pytest_cache/",
                ".venv/",
                ".env",
                ".runs/",
                "",
            ]
        ),
        project_root / ".env.example": "\n".join(
            [
                "OPENAI_API_KEY=",
                "ANTHROPIC_API_KEY=",
                "GOOGLE_API_KEY=",
                "LOCAL_MODEL_PROVIDER=ollama",
                "OLLAMA_BASE_URL=http://localhost:11434",
                "DEFAULT_LOCAL_MODEL=gemma3:4b",
                "AGENTMESH_RUN_DIR=.runs",
                "",
            ]
        ),
        project_root / "requirements.txt": "\n".join(
            [
                "agentmesh",
                "fastapi>=0.110",
                "pydantic>=2.7",
                "pyyaml>=6.0",
                "rich>=13.7,<14",
                "typer>=0.12",
                "uvicorn>=0.29",
                "",
            ]
        ),
        project_root / "project.yaml": yaml.safe_dump(project_spec(project), sort_keys=False),
        paths["personas"] / "user_persona.md": "# User Persona\n\nDescribe the target user here.\n",
        paths["personas"] / "use_case_persona.md": f"# Use Case Persona\n\n{project.description}\n",
        paths["models"] / "model_routing.yaml": yaml.safe_dump(
            {
                "models": {
                    "local/gemma-3-4b": {
                        "provider": "local",
                        "cost_per_1k_input_tokens": 0,
                        "cost_per_1k_output_tokens": 0,
                    },
                    "openai/gpt-4.1-mini": {
                        "provider": "openai",
                        "cost_per_1k_input_tokens": 0.0004,
                        "cost_per_1k_output_tokens": 0.0016,
                    },
                    "google/gemini-flash": {
                        "provider": "google",
                        "cost_per_1k_input_tokens": 0.0005,
                        "cost_per_1k_output_tokens": 0.0015,
                    },
                },
                "routing_rules": {"default_local": "local/gemma-3-4b"},
            },
            sort_keys=False,
        ),
        paths["response_contracts"] / "response.yaml": yaml.safe_dump(
            {
                "format": "markdown",
                "required_fields": ["summary", "findings", "recommendations"],
            },
            sort_keys=False,
        ),
        agents_dir / f"{project.id}_agents.py": executor_source(project),
    }

    for agent in project.agents:
        files[paths["agents"] / f"{agent.id}.yaml"] = yaml.safe_dump(
            agent_spec(agent),
            sort_keys=False,
        )

    agents_dir.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        path.write_text(content, encoding="utf-8")
        written.append(path)

    if create_venv:
        venv_path = project_root / ".venv"
        if venv_path.exists() and not overwrite:
            raise FileExistsError(f"Virtual environment already exists: {venv_path}")
        venv.EnvBuilder(with_pip=True).create(venv_path)
        written.append(venv_path)

    return written
