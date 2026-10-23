# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: **Security -> Report a vulnerability** on this
repository (private advisory). Do not open a public issue. We acknowledge within 2 working days and aim to
ship a fix or mitigation within 7 days for high-severity issues.

## Supported versions

Only the release currently deployed on Liara (the newest `v*` tag) receives fixes.

## How this repository protects secrets and the supply chain

- No secret is ever committed: every push and pull request runs **gitleaks** over the full history, and
  `.gitignore` excludes `.env*`, keys, databases and private notes.
- Runtime secrets live only in the Liara app environment (set via the console or
  `POST /v1/projects/update-envs`) and in root-only (0600) files on the operator hosts. Deploy credentials
  for CI are GitHub Actions secrets (`LIARA_API_TOKEN`, `LIARA_TEAM_ID`); release bodies never carry env values.
- Logs and tool output pass through redaction (tokens, JWTs, bearer values) before they are printed.
- Dependencies are pinned; CI runs **pip-audit**/**npm audit**, builds an **SBOM** (SPDX + CycloneDX) and a
  **Trivy** image scan on every change; **Dependabot** proposes updates weekly.
- Third-party GitHub Actions are pinned to full commit SHAs; workflows run with `contents: read`.
- Containers run as an unprivileged user where the image supports it and ship no VCS metadata or env files
  (checked in CI).

Further details: [docs/SECURITY.md](docs/SECURITY.md) (application-level controls) and
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) (pipeline).
