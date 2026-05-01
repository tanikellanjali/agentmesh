from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentmesh import __version__
from agentmesh.core.project_loader import list_projects, load_project
from agentmesh.core.runtime import run_project
from agentmesh.core.spec_loader import SpecLoadError

app = FastAPI(title="AgentMesh API", version=__version__)


class RunRequest(BaseModel):
    message: str
    user_id: str = "default_user"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/projects")
def projects() -> list[dict[str, str]]:
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
def project_agents(project_id: str) -> list[dict[str, object]]:
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
def run_project_endpoint(project_id: str, request: RunRequest) -> dict[str, object]:
    try:
        result = run_project(project_id, request.message)
    except SpecLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = result.as_dict()
    response["user_id"] = request.user_id
    return response
