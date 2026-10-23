# Operations

Day-2 guide for whoever is on call while the engine runs. Production is the Liara app (section 0); sections 1–5
cover the optional self-hosted systemd deployment. Installation and upgrades are in
[DEPLOYMENT.md](DEPLOYMENT.md); this document covers watching, reacting and reporting.

## 0. Liara production (`meowmeow-wallex`)

### Emergency stop

1. Open `https://meowmeow-wallex.liara.run/live` → **DISARM now** (no password needed). Takes effect on the next order.
2. If the page is unreachable: `python3 deploy/liara-wallex/deploy_liara.py setenv WALLEX_LIVE_ENABLED=0`
   (restart recreates the kill file), or turn the app off in the Liara console
   (`POST /v1/projects/meowmeow-wallex/actions/scale {"scale":0}`).
3. Flatten open positions in the Wallex UI if required (`cancel()` is refused while disarmed; with the app off the
   watchdog is off too).

### What to watch

| Signal | Where | Healthy / alert |
|---|---|---|
| Liveness | `GET /healthz` (no auth) | `ok`; alert if non-200 for > 2 min |
| Arm state | `/api/live` → `state` | `armed` matches the plan; alert on unexpected `true` |
| Engine process | `/api/live` → `state.engine_process` | `running: true` with stable `starts`; alert if `starts` keeps rising (restart loop) |
| Self-check | `/api/live` → `selfcheck` | Wallex 200, `boxapi_models` ok, `global_source` not null |
| Errors / decisions | `/live` page | No repeating errors; AI decisions arriving |
| Container events | `GET /v1/projects/meowmeow-wallex/events?start=..&end=..` | Alert on `oom`, `die`, `kill` without a deploy |
| CPU / memory / disk | `GET /v1/projects/meowmeow-wallex/metrics/summary` | Memory well under the plan; disk `wallex-data` < 80 % |
| Logs | `make liara-logs` or `GET /v2/projects/meowmeow-wallex/logs` | `[start] selfcheck`, engine output; no tracebacks |
| BoxAPI spend | `/live` decisions (`usd`), per-account `llm-budget.sqlite` on `/data` | Under the daily cap |

A minimal external check (cron on any host with the admin password in a 0600 file):

```bash
P=$(cat ~/.wallex-admin-password)   # 0600
curl -fsS https://meowmeow-wallex.liara.run/healthz >/dev/null || echo "wallex: healthz failed"
curl -fsS -u admin:"$P" https://meowmeow-wallex.liara.run/api/live | python3 -c "
import json,sys; s=json.load(sys.stdin); st=s['state']
print('armed' if st['armed'] else 'disarmed', 'engine', st['engine_process'], 'global', s.get('selfcheck',{}).get('global_source'))"
```

### On-call checklist (Liara)

- [ ] `/healthz` ok; `/live` loads; arm state as planned; engine process stable.
- [ ] Self-check: Wallex reachable, BoxAPI ok with the expected key count, `global_source` set.
- [ ] If armed: **Read balances** matches the Wallex UI; watchdog running (engine process list); `liquidate_at_utc`
      still the official cutoff.
- [ ] After any deploy or rollback: re-check env vars (a script deploy resets them to the disarmed defaults).
- [ ] End of run: confirm TMN-only holdings, **Disarm**, set `WALLEX_ENGINE=off`, back up the `wallex-data` disk
      (console → Disks → Backups) and archive the audit logs.

The rest of this document covers the optional self-hosted deployment.

## 1. Emergency stop, self-hosted (memorise this)

```bash
sudo touch /var/lib/wallex-arena/LIVE_DISABLED     # blocks every live order immediately (or: make kill-switch)
```

- Takes effect on the next order call; no restart needed. Paper trading continues unaffected.
- While the file exists the client also refuses `cancel()`. To flatten a live account, use the Wallex web/app UI.
- To stop everything: `sudo systemctl stop wallex-arena` (positions stay open on the exchange; if armed, the
  watchdog will place exchange-side stops after 120 s — stop it too if you are flattening by hand:
  `sudo systemctl stop wallex-live-watchdog`).

## 2. What runs and how it fails

| Unit | Restart policy | If it dies | Safety net |
|---|---|---|---|
| `wallex-arena` | `always`, 20 s | No new decisions, no stop checks, `status.json` goes stale | Watchdog (live) places exchange-side stops after 120 s |
| `wallex-live-watchdog` | `always`, 10 s | No dead-man protection | Arena's own stops while it runs |
| `wallex-paper-agent` | `on-failure`, 30 s, max 4 starts / 10 min | Paper portfolio frozen | None needed (paper only) |
| `wallex-book-recorder` | `always`, 10 s | Gap in recorded market data | — |
| `wallex-dashboard-export.timer` | every 30 s | Dashboard shows stale data | `health.json.updated_at` reveals it |
| `wallex-dashboard-web` | `always`, 5 s | Dashboard offline | — |

Built-in automatic protections in the arena (no operator action needed): per-position stops and max hold,
5 % drawdown → 60 min pause, 10 % drawdown → liquidate and halt for the run, USDT/TMN anomaly halt
(`usdt_tmn_halted`), fast-drop guard that pulls bids in markets whose global price is falling (`guarded`),
no entries in the final hour, full TMN liquidation at `liquidate_at_utc`.

## 3. Monitoring

### Signals worth alerting on

| Signal | Source | Threshold | Severity |
|---|---|---|---|
| Arena heartbeat | `now - status.json.updated_at` | > 90 s | page |
| Watchdog emergency | `watchdog.json.emergency` | `true` | page |
| Live account error | `live.jsonl` entries with `AmbiguousOrder` / `WallexAPIError` | any | page |
| Kill file unexpectedly absent | `/var/lib/wallex-arena/LIVE_DISABLED` | missing while not intentionally live | page |
| Account halted | leaderboard `halted` | any account | notify |
| Unit not active | `systemctl is-active` | arena / paper agent / watchdog (if live) | page |
| Feed degraded | `status.json.stream.connected`, `feed_errors_recent` | disconnected > 5 min | notify |
| LLM spend | leaderboard `strategy.llm.last_24h_usd` vs `daily_cap_usd` | ≥ 80 % | notify |
| Paper agent errors | `daemon.json.consecutive_errors` | ≥ 3 | notify |
| Disk | `/var/lib` usage | > 80 % | notify |

### Minimal alert script (cron, every minute)

No monitoring stack is required; this cron job logs to the journal and can be wired to any notifier:

```bash
sudo tee /usr/local/bin/wallex-healthcheck >/dev/null <<'EOF'
#!/usr/bin/env bash
S=/var/lib/wallex-arena/status.json
age=$(python3 -c "import json,time;print(int(time.time()-json.load(open('$S'))['updated_at']))" 2>/dev/null || echo 9999)
[ "$age" -gt 90 ] && logger -p user.crit -t wallex "arena status stale: ${age}s"
for u in wallex-arena wallex-paper-agent; do systemctl is-active --quiet $u || logger -p user.crit -t wallex "$u not active"; done
grep -q '"emergency": true' /var/lib/wallex-arena/watchdog.json 2>/dev/null && logger -p user.crit -t wallex "watchdog emergency"
exit 0
EOF
sudo chmod 755 /usr/local/bin/wallex-healthcheck
echo '* * * * * root /usr/local/bin/wallex-healthcheck' | sudo tee /etc/cron.d/wallex-healthcheck
journalctl -t wallex -f      # watch alerts
```

### Dashboard

`wallex-dashboard-web` serves a password-protected page (leaderboard, equity curves, recent trades, feed health) on
`https://<DASHBOARD_HOST>:9445/`, refreshed every 30 s by `wallex-dashboard-export.timer`. User `meowmeow`,
password in `/etc/wallex/dashboard-password.txt`. Raw JSON: `/data/{leaderboard,equity,trades,health}.json`.

## 4. On-call checklist

**Start of shift**

- [ ] `make status` — expected units active; no restart loops (`systemctl show wallex-arena -p NRestarts`).
- [ ] Heartbeat < 60 s; `stream.connected` true; `usdt_tmn_halted` false.
- [ ] Kill switch state matches the plan (present = paper only).
- [ ] If live: watchdog active and `emergency: false`; Wallex UI balances match `make arena-status` for live accounts.
- [ ] LLM spend within caps; disk < 80 %; last backup < 1 h old (`ls /var/backups/wallex`).
- [ ] Confirm `liquidate_at_utc` in the config is still the official cutoff.

**During incidents**

1. Real money at risk or behaviour you cannot explain → **kill switch first**, investigate second.
2. Capture evidence: `cp -a /var/lib/wallex-arena/status.json /tmp/incident-$(date -u +%s).json`;
   `journalctl -u wallex-arena --since -30min > /tmp/arena.log`.
3. Check the account's `decisions.jsonl` tail for the last decisions and their reasons.
4. Fix, then restart the affected unit; verify heartbeat and leaderboard.
5. Remove the kill file only after a second person agrees.
6. Write a short note: time, symptom, action, outcome.

**End of run (challenge cutoff)**

- [ ] After `liquidate_at_utc`, confirm all accounts hold only TMN (`make arena-status`), live account also in the UI.
- [ ] Create the kill file; stop the arena.
- [ ] `sudo -u wallex .venv/bin/python -m experiments.wallex_arena report /var/lib/wallex-arena /tmp/report.md`.
- [ ] Verify every audit chain (`DecisionLog.verify`) and archive a backup off-host.

## 5. Routine tasks

| Task | Command |
|---|---|
| Leaderboard | `sudo -u wallex /opt/wallex-trading/.venv/bin/python -m experiments.wallex_arena status /var/lib/wallex-arena` |
| Shadow-test report | `sudo -u wallex ... -m experiments.wallex_arena report /var/lib/wallex-arena /tmp/report.md` |
| Paper agent status | `sudo -u wallex ... -m experiments.wallex_agent status --state /var/lib/wallex-paper-agent` |
| Change strategy params | edit `arena-config.json` → `sudo systemctl restart wallex-arena` (accounts keep state by name) |
| Add an excluded market | edit `experiments/wallex_arena/excluded_markets.json` → restart arena and recorder |
| Rotate a key | edit `/etc/wallex/<file>.env` → restart the units that load it |
| Backup now | `sudo /opt/wallex-trading/scripts/backup_state.sh` |
| Recorder DB size | `du -h /var/lib/wallex-book-recorder/books.sqlite` (rotate by stopping the unit and moving the file) |

<!-- platform:begin (maintained by the platform kit) -->
## Platform operations: SLOs, backups, runbooks

### SLOs and monitoring

**SLOs** (30-day rolling): availability **≥ 99.5 %** and p95 latency **< 1.5 s** on `/healthz`.

**Monitoring (live).** `meow-monitor.timer` on Hetzner (Helsinki) probes every 60 s:

- status, content, latency with DNS excluded, and TLS expiry;
- a failed probe is retried once after 5 s;
- the result is cross-checked with Negahban's in-Liara probes, and the app counts as down only when both
  vantages fail;
- a run where every app fails from Helsinki is a probe-side `vantage_unreachable` note, not an outage.

**Alerts.** An app is marked DOWN after 2 consecutive down verdicts; alerts go to a JSON log and an optional
webhook.

**Status page:** https://negahban-sre.liara.run/status shows both vantages for each app.

**Error-budget policy:**

- While more than 50 % of the budget remains, deploy freely.
- Below 50 %, deploy only with a reviewer.
- When the budget is exhausted, freeze features and allow only fixes and rollbacks.

### Backups and restore

The `wallex-data` disk (1 GB) holds the engine state, decision logs and kill file. The bundle is basic, so there are no native Liara backups. `meow-backup.timer` (daily 02:30 UTC, Hetzner) exports it through the Liara file browser API and verifies it (sha256, SQLite integrity, JSON parse). It keeps 7 daily + 4 weekly copies. **Status: ready, not enabled.** Enable with `systemctl enable --now meow-backup.timer`. Restore: `liara_backup.py verify`, then `restore --app meowmeow-wallex --disk wallex-data` (dry run, then `--yes`), then swap the files in with the engine disarmed and the app scaled to 0. See the platform document §8.

### Runbooks

Full commands: platform document §10. Shortest paths for Wallex:

| # | Symptom | First action |
|---|---|---|
| R1 | Errors right after a release / Deploy exit 3 | `python3 infra/liara_deploy.py rollback --manifest infra/liara.app.yaml` then `... smoke` |
| R2 | Probe DOWN, 502/503, crash loop | Read events (`oom`/`die`) and logs (`GET /v2/projects/meowmeow-wallex/logs`), then `POST /v1/projects/meowmeow-wallex/actions/restart`. For OOM, raise `plan` within `scaling.max`, then `liara_apply.py plan` / `apply` |
| R3 | Broke after an env change | `liara_apply.py plan --app meowmeow-wallex` (missing names), then roll back to the previous ENV_UPDATE release (`rollback --to <id>`) |
| R4 | LLM errors / timeouts | Keys rotate automatically; switch `LLM_PROVIDER`; Arvan gate kill switch `sudo touch /etc/nginx/meow-llm.disabled` |
| R5 | Disk full or data damaged | Read `metrics/summary` → `disksUsage`; grow `sizeGB` and apply; or scale 0, restore (above), scale 1 |
| R6 | `frozen_project` / low credit | `liara_apply.py plan` (runway), top up, `POST /v1/projects/meowmeow-wallex/unfreeze` |

### On-call checklist (platform)

1. Check the status page and the last CI/Deploy run.
2. Check `liara_deploy.py status`. A release in the last hour points to R1.
3. Mitigate first (rollback / restart / provider switch), then debug.
4. Verify with `liara_deploy.py smoke` and confirm the probe is UP.
5. File an *Incident / postmortem* issue: UTC timeline, cause, actions.
<!-- platform:end -->
