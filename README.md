# AgentMesh

AgentMesh is a spec-driven open-source runtime for building internal agent meshes.

The core idea is:

```text
user request -> required capabilities -> matching YAML agents -> internal Python executors -> structured outputs
```

AgentMesh is not API-first. The API and CLI are entry points into the same internal runtime. Agents are loaded from specs and executed by Python code inside the orchestrator.

## What Works Today

- Python package skeleton
- Typer CLI
- FastAPI wrapper endpoints
- YAML project and agent spec loading
- In-memory agent registry
- Rule-based need resolver
- Capability matcher
- Internal orchestrator
- Generic deterministic Python agent executor
- `data_ops` example spec pack
- Deterministic project generator for new starter packs
- Tests for spec loading, matching, runtime execution, and project generation

## What Is Not Built Yet

- LLM-based project generation from a user description
- Rich editing flow for generated agent specs/scripts
- Real model provider calls
- Run logs under `.runs/`
- Plugin installer
- Model router/provider availability checks

Those are planned open-source features, but the current codebase does not fully do them yet.

## Quick Start

```bash
pip install -e ".[dev]"
agentmesh --help
agentmesh help-functions
agentmesh list-projects
agentmesh list-agents --project data_ops
agentmesh init-project "Build a customer support triage mesh"
agentmesh configure-models
agentmesh list-models --project data_ops
```

## Open Source Project Files

This repository includes:

- `LICENSE`: MIT license.
- `CONTRIBUTING.md`: contribution setup and project rules.
- `CODE_OF_CONDUCT.md`: community expectations.
- `SECURITY.md`: vulnerability and secret-handling guidance.
- `docs/roadmap.md`: planned milestones.
- `docs/release_checklist.md`: release steps.
- `examples/`: sample requests.

To generate a project scaffold with an actual virtual environment:

```bash
agentmesh init-project "Build a customer support triage mesh" --create-venv
```

Run the current `data_ops` example pipeline:

```bash
agentmesh run --project data_ops "Review this SQL query for performance, missing values, and summarize the risks."
```

Start the API wrapper:

```bash
uvicorn agentmesh.api.server:app --reload
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/projects/data_ops/run \
  -H "Content-Type: application/json" \
  -d '{"message":"Review this SQL query for performance, missing values, and summarize the risks."}'
```

## How To Add Agents Manually Today

1. Add a YAML spec in `spec_packs/<project>/agents/<agent_id>.yaml`.
2. Give it capabilities such as `sql_analysis` or `report_generation`.
3. Add or update keywords in `agentmesh/core/need_resolver.py`.
4. Create a Python executor in `agentmesh/agents/`.
5. Register that executor in `agentmesh/agents/factory.py`.
6. Run `pytest`.

See [docs/open_source_usage.md](docs/open_source_usage.md) for the intended creator workflow.

## Model Plugins

The current runtime uses deterministic local Python executors. It does not call model providers yet.

The intended model/provider plugin list is documented in [docs/model_plugins.md](docs/model_plugins.md).
Cost estimation is documented in [docs/cost_estimation.md](docs/cost_estimation.md).

Optional AgentFlow tool packages can be installed with:

```bash
pip install -e ".[agentflow]"
```
