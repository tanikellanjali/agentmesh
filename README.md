# AgentMesh

AgentMesh is a spec-driven open-source runtime for building internal agent meshes.

The core idea is:

```text
user request -> required capabilities -> matching YAML agents -> internal Python executors -> structured outputs
```

AgentMesh is not API-first. The API and CLI are entry points into the same internal runtime. Agents are loaded from specs and executed by Python code inside the orchestrator.

## What Works Today

- Spec-driven runtime: YAML agents, Pydantic validation, in-memory registry
- **Dynamic agents** - capabilities no registered agent covers are synthesized at
  runtime, run as ephemeral agents, and are only written to disk when you ask
- **Dependency graph** - agents declare `depends_on`, run in topological levels,
  and independent agents in a level execute concurrently
- **Real model calls** - Anthropic (Claude) and OpenAI adapters behind one
  provider interface, plus a deterministic mock provider for offline work
- **Retries** with exponential backoff and jitter on transient failures only
- **Telemetry** - tokens, cost, attempts, duration, and outcome recorded for
  every model call, with human or JSON logs
- **Opt-in budget caps** - no default ceiling; you set one or there isn't one
- Executor registry, output-contract validation, Typer CLI, FastAPI wrapper

## What Is Not Built Yet

- LLM-based agent synthesis (synthesis is deterministic today)
- Run logs persisted under `.runs/`
- Ollama / local provider adapter
- Per-call timeouts
- API authentication and rate limiting

## Quick Start

```bash
pip install -e ".[dev]"

# Install the providers you intend to use
pip install -e ".[anthropic]"   # Claude
pip install -e ".[openai]"      # OpenAI
pip install -e ".[providers]"   # both

agentmesh --help
agentmesh help-functions
agentmesh list-projects
agentmesh list-agents --project data_ops
```

### Choose a model and supply credentials

```bash
agentmesh configure-models
```

This prompts for the API keys you have and, optionally, a default provider and
model. It writes `.env`, **merging** with what is already there - re-running it to
add one provider will not wipe another's key. `.env` is gitignored.

```bash
agentmesh list-models --project data_ops   # shows which providers are usable
```

Credentials are read from `.env` and the environment. Agents pick their model from
their own spec (`models.default`, then `models.fallback`); `--provider` and
`--model` override that for a whole run.

### Run

```bash
# Offline - no credentials needed, deterministic output
agentmesh run --project data_ops "Review this SQL query for quality risks" --provider mock

# With a real model
agentmesh run --project data_ops "Review this SQL query for quality risks"

# Force one model for the whole run
agentmesh run --project data_ops "..." --provider anthropic --model anthropic/claude-sonnet-5
```

If no provider is usable, the run fails with an explicit error rather than
silently returning placeholder output.

### Dynamic agents

When a request needs a capability no registered agent covers, AgentMesh
synthesizes an agent for it and runs it. Synthesized agents are **ephemeral** -
they vanish when the run ends:

```bash
agentmesh run --project data_ops "Research the options and summarize them."
```

Keep the ones worth keeping:

```bash
agentmesh run --project data_ops "Research the options and summarize them." --save-agents
```

Saved agents are written to the project's `agents/` directory as ordinary specs,
so the next run loads them from disk and synthesizes nothing. Use
`--no-synthesize` to fall back to the troubleshooting agent instead.

### Retries, cost, and budget

Every model call is retried on transient failures - 5xx responses, connection
errors, and rate limits. Auth failures and refusals are never retried.

```bash
agentmesh run --project data_ops "..." --max-attempts 5   # default 3; 1 disables
```

Each run reports tokens, cost, calls, and retries per agent:

```
┃ Agent              ┃ Model                   ┃ Calls ┃ Retries ┃  In ┃ Out ┃ Cost $ ┃
│ data_quality_agent │ anthropic/claude-opus-5 │     1 │       0 │  23 │  30 │ 0.0009 │
│ sql_analyzer_agent │ anthropic/claude-opus-5 │     1 │       0 │  23 │  30 │ 0.0009 │
│ report_agent       │ anthropic/claude-opus-5 │     1 │       0 │ 208 │  30 │ 0.0018 │
│ total              │                         │     3 │       0 │ 254 │  90 │ 0.0035 │
```

For machine-readable telemetry - one JSON object per model call, with
`run_id`, `call_id`, `attempts`, `outcome`, tokens, cost and duration:

```bash
agentmesh run --project data_ops "..." --json-logs --log-level INFO
```

**There is no default spending cap.** AgentMesh will not decide your budget for
you. Set one when you want one:

```bash
agentmesh run --project data_ops "..." --max-cost 0.50
```

The cap blocks *new* calls once it is reached. Agents running concurrently in the
same level can overshoot by up to one level's spend; add `--max-workers 1` for a
strict ceiling. Exit code `2` means the run stopped on budget.

### Verify your credentials

```bash
agentmesh check-models --project data_ops          # can each provider be constructed?
agentmesh check-models --project data_ops --live   # send one tiny real request each
```

`--live` bills your account for a few tokens per provider.

### The agent graph

Agents run in dependency levels, not a fixed order. An agent declares what it
needs:

```yaml
depends_on:
  - ingestion_agent
```

An agent with the `response_composition` capability implicitly waits for every
other agent and writes the final response - so the answer comes from the composer,
not from whichever agent happened to finish last. Explicit `depends_on` always
wins, and cycles or unknown dependencies fail loudly at graph-build time.

### Start the API

```bash
uvicorn agentmesh.api.server:app --reload

curl -X POST http://127.0.0.1:8000/projects/data_ops/run \
  -H "Content-Type: application/json" \
  -d '{"message":"Review this SQL query.","provider":"mock","store_agents":false}'
```

> The API has no authentication yet. Do not expose it publicly.

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

## How To Add Agents

Most agents need no Python at all - a YAML spec with an `output_contract` runs on
the generic model-backed executor:

1. Add `spec_packs/<project>/agents/<agent_id>.yaml`.
2. Give it `capabilities`, an `output_contract`, and a `models.default` routing key.
3. Add keywords for those capabilities in `agentmesh/core/need_resolver.py` so
   requests route to it.
4. Run `pytest`.

For bespoke behaviour, write an executor and point the spec at it:

```python
from agentmesh.agents.base_agent import BaseAgent
from agentmesh.agents.registry import register_executor

@register_executor("sql_analyzer")
class SqlAnalyzerAgent(BaseAgent):
    def run(self, message, context):
        ...
```

```yaml
executor: sql_analyzer
```

See [docs/open_source_usage.md](docs/open_source_usage.md) for the creator workflow.

## Model Plugins

Built-in providers: `anthropic`, `openai`, and `mock` (`local` is currently an
alias for `mock` until an Ollama adapter lands). Register your own:

```python
from agentmesh.providers import register_provider

register_provider("my-provider", MyProvider)
```

A provider implements one method - `complete(system, prompt, model, max_tokens, effort)`
returning a `Completion` with `usage`. See `agentmesh/providers/base.py`.

Model routing lives in each project's `models/model_routing.yaml`, where a stable
routing key maps to a provider, a real model id, and per-token cost:

```yaml
models:
  anthropic/claude-opus-5:
    provider: anthropic
    model: claude-opus-5
    cost_per_1k_input_tokens: 0.005
    cost_per_1k_output_tokens: 0.025
```

Cost estimation is documented in [docs/cost_estimation.md](docs/cost_estimation.md).
The intended plugin list is in [docs/model_plugins.md](docs/model_plugins.md).
