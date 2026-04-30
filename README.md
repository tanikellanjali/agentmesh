# AgentMesh

AgentMesh is a spec-driven, CLI/API-first runtime for open-source agent mesh projects.

The current implementation covers the first two MVP milestones:

- package skeleton
- FastAPI health and project/agent endpoints
- Typer CLI
- YAML project and agent spec loading
- in-memory agent registry
- generic `data_ops` spec pack

## Quick Start

```bash
pip install -e ".[dev]"
agentmesh --help
agentmesh list-projects
agentmesh list-agents --project data_ops
uvicorn agentmesh.api.server:app --reload
```

Optional AgentFlow tool packages can be installed with:

```bash
pip install -e ".[agentflow]"
```
