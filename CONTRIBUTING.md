# Contributing to AgentMesh

Thanks for helping build AgentMesh.

AgentMesh is early-stage. The project is currently focused on a spec-first,
CLI/API-accessible runtime for generating and running internal agent meshes.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Useful Commands

```bash
agentmesh help-functions
agentmesh list-projects
agentmesh validate-spec --project travel_planner
agentmesh run --project travel_planner "Plan a weekend around New Jersey"
```

## Contribution Guidelines

- Keep the runtime spec-first.
- Do not make the API own agent execution.
- Keep agents discoverable through YAML specs.
- Keep provider calls behind model routing/provider interfaces.
- Add tests for new runtime behavior.
- Avoid committing secrets, `.env`, `.venv`, generated runs, or API keys.

## Good First Areas

- Model router and provider interfaces.
- `.runs/` logging and saved artifacts.
- Project generator improvements.
- Tool adapter execution.
- Better docs and examples.
