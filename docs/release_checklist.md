# Release Checklist

Use this checklist before publishing an AgentMesh release.

## Local Checks

```bash
pytest
agentmesh help-functions
agentmesh list-projects
agentmesh validate-spec --project data_ops
agentmesh run --project data_ops "Review this SQL query for quality risks"
```

## Repository Checks

- Confirm `LICENSE` exists.
- Confirm `README.md` has install and quick-start instructions.
- Confirm no secrets are staged.
- Confirm `.env`, `.venv`, `.runs`, and `__pycache__` are ignored.
- Confirm examples run.
- Confirm docs are updated.

## Git Commands

```bash
git status
git add .
git commit -m "Prepare AgentMesh open-source MVP"
git push
```

## Tag

```bash
git tag v0.1.0
git push origin v0.1.0
```
