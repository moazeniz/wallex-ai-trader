## What and why

<!-- One paragraph: the change and the reason. Link the issue. -->

## Risk and rollout

- [ ] Low risk / config or docs only
- [ ] Touches a release path (Dockerfile, infra/, liara.json, start-up, migrations): I know the rollback (`python infra/liara_deploy.py rollback`)
- Rollout: merge -> CI green -> tag `vX.Y.Z` -> Deploy workflow (smoke tests + automatic rollback)

## Checklist

- [ ] Tests added/updated and `CI` is green (lint, tests, image build, gitleaks, dependency audit)
- [ ] No secrets, `.env`, databases or personal data in the diff (gitleaks enforces this)
- [ ] New env var? Name added to `infra/*.yaml` (`envs.required`/`optional`); value set in Liara, not in git
- [ ] Docs updated (README / docs/DEPLOYMENT.md / docs/OPERATIONS.md) when behaviour or operations change
