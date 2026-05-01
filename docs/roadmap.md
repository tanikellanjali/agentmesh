# Roadmap

## v0.1 MVP

- Spec loading from YAML.
- CLI commands for listing, validating, and running projects.
- FastAPI wrapper around the internal runtime.
- Need resolution and capability matching.
- Internal agent execution.
- Data ops example pack.
- Deterministic project generator.
- Model configuration prompts and model listing.

## Next

- Generate richer agent plans from user descriptions.
- Let users edit/confirm agent plans before writing files.
- Register generated Python executors automatically.
- Write `.runs/<run_id>/` artifacts:
  - request
  - selected capabilities
  - graph
  - events
  - agent outputs
  - final response
  - cost summary
- Add model router and provider interfaces.
- Add local mock provider.
- Add Ollama provider.
- Add OpenAI, Anthropic, and Gemini provider adapters.
- Add tool adapter interfaces.
- Add `agentmesh logs` and `agentmesh explain-failure`.

## Later

- Plugin installer.
- Better graph builder.
- Cost estimator before execution.
- Response contract validator.
- Learning-loop suggestions that do not auto-edit specs.
- PyPI publishing.
- GitHub Actions CI.
