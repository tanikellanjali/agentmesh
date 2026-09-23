from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from agentmesh import __version__
from agentmesh.access.principal import Principal
from agentmesh.api.auth import Authenticator
from agentmesh.core.project_loader import list_projects, load_project
from agentmesh.core.runtime import run_project
from agentmesh.core.spec_loader import SpecLoadError
from agentmesh.core.telemetry import BudgetExceeded
from agentmesh.providers import ProviderError
from agentmesh.providers.retry import RetryPolicy

app = FastAPI(title="AgentMesh API", version=__version__)
authenticator = Authenticator()


def current_principal(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> Principal:
    """Authenticate the caller and hand back who they are."""
    presented = x_api_key
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization.split(" ", 1)[1].strip()

    try:
        return authenticator.authenticate(
            presented, request.client.host if request.client else None
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


class RunRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    synthesize: bool = True
    store_agents: bool = False
    provider: str | None = None
    model: str | None = None
    max_attempts: int = 3
    max_cost: float | None = None


@app.get("/health")
def health() -> dict[str, object]:
    """Unauthenticated: a liveness probe must not need a credential."""
    return {"status": "ok", "version": __version__, "auth_configured": authenticator.configured}


@app.get("/projects")
def projects(principal: Principal = Depends(current_principal)) -> list[dict[str, str]]:
    return [
        {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "entrypoint": project.entrypoint,
        }
        for project in list_projects()
    ]


@app.get("/projects/{project_id}/agents")
def project_agents(
    project_id: str, principal: Principal = Depends(current_principal)
) -> list[dict[str, object]]:
    try:
        loaded = load_project(project_id)
    except SpecLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return [
        {
            "id": agent.id,
            "name": agent.name,
            "category": agent.category,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "enabled": agent.enabled,
        }
        for agent in loaded.agents
        if agent.enabled
    ]


@app.post("/projects/{project_id}/run")
def run_project_endpoint(
    project_id: str,
    request: RunRequest,
    principal: Principal = Depends(current_principal),
) -> dict[str, object]:
    try:
        result = run_project(
            project_id,
            request.message,
            synthesize=request.synthesize,
            store_agents=request.store_agents,
            provider=request.provider,
            model=request.model,
            max_cost=request.max_cost,
            retry_policy=RetryPolicy(max_attempts=request.max_attempts),
            principal=principal,
        )
    except SpecLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BudgetExceeded as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = result.as_dict()
    response["user_id"] = request.user_id
    return response
