from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelDefinition] = Field(default_factory=dict)
    routing_rules: dict[str, Any] = Field(default_factory=dict)
