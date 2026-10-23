# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/),
versioning: [SemVer](https://semver.org/).

## [1.2.0] - 2026-10-22

Live on Liara (`meowmeow-wallex`): the `liara-live` branch up to 3eede15, applied on top of 1.1.0.

### Added
- `LLM_PROVIDER=anthropic|boxapi` switch with fallback on 429/5xx/credit/auth/network errors
  (`experiments/wallex_arena/anthropic_http.py`). Anthropic is reached over plain HTTP through the team LLM gate
  (`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` hold the gate URL and gate token). Model tiers are mapped across
  providers (gpt-6.1-sol <-> claude-sonnet-5-5, gpt-6-luna <-> claude-haiku-5-5), and decisions use native
  structured output.
- Observability and resilience: JSON-lines logs with request/trace IDs (`obs.py`); one retry helper with full-jitter
  backoff and circuit breakers (`resilience.py`; orders are never retried automatically); a durable job queue with a
  dead letter and admin endpoints (`jobqueue.py`, `/api/jobs`); `/readyz`; a global error handler; graceful shutdown.
- In-image smoke tests (`deploy/liara-wallex/test_liara_image.py`), run during the image build.

### Changed
- The container runs as uid 10001 (`setpriv` after a one-time `chown` of `/data`). Engine state, kill file and job
  queue live under `WALLEX_STATE_DIR=/data/wallex-arena`.
- `WALLEX_MODEL` is superseded by `LLM_PROVIDER` plus `WALLEX_ANTHROPIC_MODEL` / `WALLEX_BOXAPI_MODEL`.
- A code-only image (repository-root Dockerfile without `./snapshot`) reports ready; only a bundled-but-incomplete
  results snapshot makes `/readyz` fail.

## [1.1.0] - 2026-10-22

Liara production release (`meowmeow-wallex`): the `liara-live` branch (c2b8f2a, based on 9c4a774) applied on top of
1.0.0, which already contains the later USDC maker commit (12daf5f).

### Added
- BoxAPI provider for AI accounts (`experiments/wallex_arena/boxapi.py`): `openai/...` models, round-robin key
  rotation with fail-over and cooldown, strict JSON-schema decisions, billed cost in the budget ledger.
- `WALLEX_GLOBAL_REST`: Binance-compatible global price source for Iran-hosted runtimes (Binance is blocked on Liara).
- Liara app (`deploy/liara-wallex/`): password-protected dashboard, `/live` control page with ARM/DISARM, boot
  connectivity self-check, BoxAPI self-test, engine and watchdog idle/disarmed by default; `deploy_liara.py`
  (`build`, `assemble`, `deploy`, `setenv`, `releases`, `rollback`), `make_snapshot.py`.
- Repository-root `Dockerfile`, `liara.json`, `.liaraignore`, `.dockerignore`; Makefile targets `docker-build`,
  `docker-run`, `liara-deploy`, `liara-deploy-api`, `liara-releases`, `liara-rollback`, `liara-logs`; CI image build
  and smoke test.

### Changed
- Docs: Liara is the primary deployment (DEPLOYMENT Part A); the Ubuntu/systemd setup is the optional self-hosted
  alternative. README, ARCHITECTURE, OPERATIONS, SECURITY updated; Anthropic marked legacy (self-hosted paper only).
- `anthropic` is now an optional import in the arena; the Liara image does not ship it.
- `deploy_liara.py`: paths and app name configurable via environment; the Liara API helper can be `lia.py` or
  `liara_cli.py`; host-specific IPs removed from `start.py` / `make_snapshot.py`.

### Fixed
- Liara image: include `experiments/wallex_agent/news.py`, which the engine's `NewsFeed` imports (the engine failed
  with `ModuleNotFoundError` when started in a build without it).
- Dashboard `/` no longer fails when the build has no results snapshot.

## [1.0.0] - 2026-10-22

Hackathon release (Starcoach Agentic AI hackathon, Wallex challenge).

### Added
- Wallex arena engine: shared live feed (Wallex + Binance WebSocket with REST fallback), rule strategies
  (`LagReversion`, `FairValueMaker` with downtrend filter, `HoldBenchmark`) and LLM strategies (`AITrader`, `AIMaker`).
- Deterministic risk envelope, depth-walking paper fills, hash-chained decision logs, Markdown shadow-test report.
- Live Wallex client (`LiveExecutor`) behind config + environment + kill-file switches; `dry_run` mode;
  dead-man watchdog with exchange-side stops.
- Paper research agent: frozen 16-market XGBoost/Ridge model, point-in-time news, three-role Haiku council,
  protective monitor, LLM budget ledger.
- Runtime services: public order-book recorder, dashboard exporter, HTTPS dashboard server and UI.
- Templated, hardened systemd units with an installer; live-mode drop-ins; state backup script.
- Docs: architecture, deployment runbooks with go-live checklist, operations/on-call, security.
- CI: lint, offline tests, unit rendering, secret scan.

### Changed
- Repository restructured for publication: runtime scripts moved to `services/`, unit files templated under
  `deploy/systemd/`, docs consolidated under `docs/`, dependencies pinned in `requirements.txt`
  (legacy framework deps in `requirements-legacy.txt`).

### Removed
- Research notes, backtest outputs, datasets, runtime state and planning documents (kept out of the repository).
