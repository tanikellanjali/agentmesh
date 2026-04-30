from pydantic import BaseModel, ConfigDict


class DefaultPersona(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    user: str
    use_case: str


class ProjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    name: str
    description: str
    entrypoint: str
    default_persona: DefaultPersona
    agent_dir: str
    model_routing: str
    response_contract: str
