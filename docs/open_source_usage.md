# Open Source Usage

This document describes how AgentMesh should be used by open-source users and what is implemented today.

## Current Runtime Flow

```text
CLI/API request
  -> core.runtime.run_project()
  -> project/spec loading
  -> need resolution
  -> capability matching
  -> orchestrator
  -> internal Python agent executors
  -> structured outputs
```

The API is only a wrapper. Agent code is called internally by the orchestrator.

## Current CLI Commands

```bash
agentmesh --help
agentmesh list-projects
agentmesh list-agents --project data_ops
agentmesh validate-spec --project data_ops
agentmesh init-project "Build a customer support triage mesh"
agentmesh run --project data_ops "Review this SQL query for quality risks"
```

## Intended Creator Flow

The open-source product should eventually support this flow:

1. User describes the system they want to build.
2. AgentMesh proposes a project description.
3. AgentMesh proposes agent descriptions:
   - agent id
   - name
   - purpose
   - capabilities
   - input contract
   - output contract
   - tools
   - preferred models
4. User confirms or edits the proposed agent plan.
5. AgentMesh generates YAML specs.
6. AgentMesh generates Python executor scripts for each agent.
7. AgentMesh runs validation.
8. AgentMesh executes the pipeline with mock/local providers first.
9. User can later plug in real models.

## Does The Current Code Do This?

Partially.

Implemented:

- Load projects and agents from YAML.
- Generate starter project scaffolds from a user description.
- Ask for confirmation before writing generated files.
- Write starter YAML specs and Python executor stubs.
- Resolve a request into capabilities.
- Match capabilities to agents.
- Execute selected agents internally.
- Run the `data_ops` example with the generic internal executor.

Not implemented yet:

- LLM-based generation of domain-specific agent plans.
- Rich interactive editing of generated plans before writing files.
- Automatic registration of domain-specific executor classes.
- Call LLM providers.

## Recommended Next Milestone

Improve the project generator:

```bash
agentmesh init-project "Build a customer support triage mesh"
```

Expected behavior:

1. Draft project spec.
2. Draft agent list.
3. Print a confirmation summary.
4. Only after confirmation, write files:
   - `spec_packs/<project>/project.yaml`
   - `spec_packs/<project>/agents/*.yaml`
   - `agentmesh/agents/<project>_agents.py`
5. Update or extend executor registration.
