from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    category: str
    description: str
    capabilities: list[str]
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]
    models: dict[str, Any]
    enabled: bool = True
    version: int = 1
    owner: str | None = None
    visibility: str = "private"  # private | team | org | public
    derived_from: str | None = None  # "<agent_id>@<version>" when adapted
    executor: str = "generic"
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    data_requirements: list[dict[str, Any]] = Field(default_factory=list)
    fallback_agents: list[str] = Field(default_factory=list)
    troubleshooting: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
