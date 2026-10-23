# Security

## Assets and threats

| Asset | Threat | Control |
|---|---|---|
| Wallex API key (can trade the account) | Leak via git, logs, shell history, chat; theft from host | Root-owned 0600 file in `/etc/wallex/`, passed by systemd `LoadCredential=` only to the units that need it; client redacts `x-api-key` in every log/repr/dry-run; `.gitignore` blocks `*.env`, `private/`, keys |
| Account funds | Runaway strategy, LLM error or prompt injection, bug, engine crash | Two-switch live gate + kill file, hard per-order limits, risk envelope, schema-validated LLM output, dead-man watchdog, credit sub-account isolation |
| BoxAPI keys (Liara) / Anthropic key (self-hosted paper only) | Leak; cost blow-up | Env vars / credential files only; key rotation with cooldown; per-account daily USD cap (arena) and a persistent attempts/spend ledger (paper agent) |
| Audit logs (judging evidence) | Tampering, loss | Hash-chained `decisions.jsonl` (`DecisionLog.verify`), backups with off-host copies |
| Dashboard | Unauthorised viewing; host compromise via web server | TLS ≥ 1.2, constant-time Basic-auth check, read-only static files, no directory listing, `DynamicUser`, empty capability set, no access log |

## Liara production specifics

- Secrets (`WALLEX_ADMIN_PASSWORD`, `BOXAPI_KEYS`, `WALLEX_API_KEY`, `WALLEX_SUBACCOUNT_ID`) exist only as Liara app
  env vars. Set them in the console or with `deploy_liara.py setenv` (values are never printed; read keys with
  `read -rs` so they do not land in shell history). Build-host secret files (`/root/.boxapi.env`,
  `/root/.wallex-liara-admin.env`, Liara token) are root-owned 0600 and never enter the image.
- The image contains code only (`.liaraignore` / `.dockerignore` exclude `.env`, `*.env`, `_private/`, `private/`,
  `*.private.md`, keys, databases and runtime state). The optional results snapshot contains paper-trading data only.
- Dashboard: every page except `/healthz` requires Basic auth (user `admin`, constant-time compare); without
  `WALLEX_ADMIN_PASSWORD` all pages are refused. POST actions require `X-Requested-With: dashboard` (CSRF guard).
  **Arm** requires the password again plus the typed phrase and all env switches; **Disarm** is always allowed.
  All control actions are appended to `control.jsonl` on the persistent disk.
- The Wallex key is copied at boot to `/run/wallex/wallex.env` (directory 0700, file 0600) inside the container.
- BoxAPI keys are rotated and redacted by `BoxAPIClient`; a failing key is cooled down rather than retried in a loop.
- Liara has no IP allow-listing on the default domain: use a long random admin password and rotate it after the event.

## Secrets handling rules (self-hosted)

1. Secrets exist only in `/etc/wallex/*.env` (root:root, 0600) on the server. Templates with placeholders are in
   `deploy/env/`. Nothing else in the repository may contain a real key.
2. Never pass a key on a command line, in an environment variable of an interactive shell, in a ticket or in chat.
   Edit files with `sudoedit`.
3. Use the **credit sub-account's own** Wallex key, restricted to what trading needs; never the main account key.
4. Rotation: create a new key in Wallex → replace it (Liara: `deploy_liara.py setenv WALLEX_API_KEY=...`, the app
   restarts; self-hosted: edit the file and `sudo systemctl restart wallex-arena wallex-live-watchdog`) → revoke the old
   key. BoxAPI: update `BOXAPI_KEYS` the same way.
5. Before every commit/push run a secret scan, e.g.
   `git grep -nE 'sk-ant-[A-Za-z0-9_-]{20,}|x-api-key: [^*]|WALLEX_API_KEY=[^R]|BEGIN (RSA|EC|OPENSSH) PRIVATE' || echo clean`
   (or `gitleaks detect`). CI can add the same check.

## Live-trading safety controls (defence in depth)

1. Config: an account trades for real only with `"live": true` (`"dry_run"` never sends).
2. Environment: `WALLEX_LIVE_ENABLED` must be exactly `1` (shipped units leave it unset / `0`).
3. Filesystem: `/var/lib/wallex-arena/LIVE_DISABLED` blocks all orders and wins over everything else.
4. Per order: TMN market, not excluded, notional ≥ 100,000 TMN and ≤ `max_order_tmn`, quantity/price rounded
   conservatively, 20 orders / 10 s throttle, create-order POST never blindly retried.
5. Portfolio: risk envelope (position/exposure caps, risk-based sizing, stops, drawdown pause/halt, final liquidation).
6. Engine failure: watchdog places exchange-side `STOP_MARKET` orders if `status.json` is stale for 120 s.

## LLM-specific risks

- News, market data and any web text placed in prompts are marked untrusted evidence, not instructions.
- Model output must parse into a strict pydantic schema; sizes, stops and holding times are clamped to allowed ranges;
  actions go through the same risk envelope as rule strategies. An LLM cannot enable live trading, change limits or
  reach the exchange directly.
- Every call is logged with request id, cost and latency for audit.

## Host hardening (as shipped in the units)

`User=` unprivileged account, `NoNewPrivileges`, `ProtectSystem=strict` with explicit `ReadWritePaths`,
`ProtectHome`, `PrivateTmp`, memory/CPU quotas; the paper agent additionally uses `ProtectKernel*`,
`RestrictAddressFamilies`, `RestrictSUIDSGID`, `LockPersonality`, `UMask=0077`, empty `CapabilityBoundingSet`.
Recommended on the host: unattended security upgrades, SSH keys only, a firewall allowing only SSH and (optionally)
the dashboard port, NTP.

## Incident response

1. Disarm: `/live` → **Disarm** on Liara, or `sudo touch /var/lib/wallex-arena/LIVE_DISABLED` self-hosted.
2. If a key may be exposed: revoke it in Wallex/BoxAPI immediately, then rotate (above).
3. Preserve evidence: backup state (`scripts/backup_state.sh`), save journals.
4. Review `live.jsonl` and the Wallex order history for unexpected orders; flatten via the Wallex UI if needed.
5. Record a timeline and the fix before re-enabling.

## Reporting

Report vulnerabilities privately through GitHub (Security → Report a vulnerability). Do not open public issues containing keys,
account identifiers or exploit details.
