# Model Plugins

AgentMesh is designed so model providers can be plugged into the runtime without agents calling providers directly.

The intended boundary is:

```text
agent executor -> model router -> provider adapter -> model API/local model
```

Agents should not call OpenAI, Anthropic, Ollama, Gemini, or any other provider directly.

## Current Status

Implemented today:

- Model config schema in `agentmesh/schemas/model_config.py`.
- YAML `models/model_routing.yaml` files in spec packs.
- Deterministic Python executors that do not require a model key.

Not implemented yet:

- `model_router.py`
- provider base interface
- local mock provider
- Ollama provider
- OpenAI provider
- Anthropic provider
- Gemini provider
- provider availability checks
- `agentmesh list-models`

## Planned Provider IDs

These model IDs are intended to work in YAML agent specs:

```yaml
models:
  default: local/gemma-3-4b
  thinking: anthropic/claude-sonnet
  fallback:
    - openai/gpt-4.1-mini
    - google/gemini-flash
    - local/gemma-3-12b
```

## Planned Providers

| Provider | Model ID Prefix | Example IDs | Required Environment |
| --- | --- | --- | --- |
| Local mock | `local/` | `local/mock`, `local/gemma-3-4b` | none |
| Ollama | `ollama/` or `local/` | `ollama/llama3.1`, `local/gemma3:4b` | `OLLAMA_BASE_URL` |
| OpenAI | `openai/` | `openai/gpt-4.1-mini`, `openai/gpt-4.1` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/` | `anthropic/claude-sonnet`, `anthropic/claude-haiku` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google/` | `google/gemini-flash`, `google/gemini-pro` | `GOOGLE_API_KEY` |

## Planned Environment Variables

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

LOCAL_MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LOCAL_MODEL=gemma3:4b

AGENTMESH_RUN_DIR=.runs
```

## Provider Selection Rules

Planned routing behavior:

1. Use `agent.models.default` when available.
2. If the task requires deeper reasoning, try `agent.models.thinking`.
3. If the selected model is unavailable, try the agent fallback list.
4. If no cloud provider key exists, use a local provider.
5. If no model is available, return a structured failure for the troubleshooting agent.

## Open-Source Plugin Shape

A provider plugin should expose:

```python
class ProviderBase:
    def is_available(self) -> bool:
        ...

    async def generate(self, payload):
        ...
```

Provider plugins should return structured model responses:

```python
{
    "text": "...",
    "model": "openai/gpt-4.1-mini",
    "provider": "openai",
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0.0,
    "raw": {}
}
```
