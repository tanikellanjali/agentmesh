# Security Policy

AgentMesh is experimental software.

## Supported Versions

Security fixes are currently targeted at the latest `main` branch until the
project starts publishing stable releases.

## Reporting a Vulnerability

Please do not publish sensitive vulnerability details in a public issue.

Open a private report through GitHub Security Advisories when enabled, or
contact the maintainers directly.

## Secret Handling

Never commit:

- `.env`
- API keys
- SSH keys
- local model credentials
- generated run data containing private prompts or outputs

The project `.gitignore` excludes common local secret and build artifacts.
