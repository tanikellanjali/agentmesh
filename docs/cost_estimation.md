# Cost Estimation

AgentMesh should show users expected run cost before they execute a generated agent mesh.

## Current Status

Implemented:

- Model routing YAML can define per-token model cost.
- `agentmesh list-models --project data_ops` shows configured models, providers, availability, and cost rates.
- `agentmesh configure-models` prompts users for provider access and writes `.env`.

Not implemented yet:

- Per-run token estimation.
- Per-agent model selection UI.
- Cost summary file under `.runs/<run_id>/cost_summary.json`.
- Actual provider usage and token accounting.

## User Setup Flow

Users should be prompted for:

1. Which providers they have access to:
   - local/Ollama
   - OpenAI
   - Anthropic
   - Google Gemini
2. API keys for the selected cloud providers.
3. Preferred local model and Ollama base URL.
4. Any model preference per agent:
   - cheap/default model
   - deeper reasoning model
   - fallback models

Current command:

```bash
agentmesh configure-models
```

Review configured models:

```bash
agentmesh list-models --project data_ops
```

## Cost Formula

For each agent:

```text
agent_cost =
  input_tokens / 1000 * cost_per_1k_input_tokens
  + output_tokens / 1000 * cost_per_1k_output_tokens
```

For a full run:

```text
total_cost = sum(agent_cost for selected agents)
```

## Example

If `sql_analyzer_agent` uses `anthropic/claude-sonnet` with:

```yaml
cost_per_1k_input_tokens: 0.003
cost_per_1k_output_tokens: 0.015
```

And the run uses:

```text
input_tokens: 3000
output_tokens: 1000
```

Estimated cost:

```text
3 * 0.003 + 1 * 0.015 = 0.024 USD
```

## Tool Declarations

Agent YAML files should not leave `tools: []` unless the agent truly has no tools.

Tools are declarative for now. The runtime does not execute tool adapters yet, but the declarations matter because they tell users what code integrations an agent expects.
