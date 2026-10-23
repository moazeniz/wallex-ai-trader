# Wallex research and paper agent

This persistent, paper-only system scans sixteen TMN markets, combines a frozen quantitative forecast with current news and three Haiku specialists, and simulates fills in one shared 5,000,000 TMN account. It contains no real order-submission transport.

## Architecture and decisions

1. `market.py` collects Wallex public metadata, closed hourly candles and bid/ask depth; missing bars are rejected rather than imputed. An optional pinned `@mathieuc/tradingview` 4.0.0 bridge supplies Binance/USDT cross-market context through the independent unofficial [TradingView-API client](https://github.com/Mathieu2301/TradingView-API). Source labels and quote units remain explicit; no USDT price becomes a TMN price.
2. `quant.py` loads frozen native XGBoost models and numeric Ridge parameters. Features need 48 uninterrupted prior hours; the target is next-open to eight-hours-later-open return. Calibration-only selection chooses an arm or cash. Forecasts reject times preceding model/calibration availability. Residual intervals are descriptive and have no coverage guarantee. No online refitting occurs.
3. `news.py` collects bounded RSS titles/summaries from crypto, central-bank, business, world/politics and Iranian economic sources. Article versions record publication, first observation and availability timestamps. A historical snapshot can use only evidence actually collected by that time; publication date alone is insufficient. Coverage is not exhaustive world news or full-article reading.
4. `runner.py` ranks opportunities by estimated TMN profit after fees, slippage, observed spread and current risk-sized budget. Held positions receive risk checks before new entries; budgets, expected-profit rankings and research targets are recomputed after those exits release capital. Up to three shortlisted new opportunities receive specialist research per cycle; held positions can receive additional research. A diagnostic shortlist is inspected when none clears the quantitative gate.
5. `council.py` asks `claude-haiku-5-5` for crypto, macro/forex/geopolitical and skeptical-risk opinions. Sources are untrusted evidence, never instructions. Directional opinions must cite supplied IDs. Confidence is an uncalibrated opinion. Semantic caching reuses unchanged evidence; model/prompt/news/bar/source changes invalidate reuse. Failed or ambiguous requests remain recorded in the budget ledger.
6. `risk.py` independently requires quantitative point edge above conservative costs plus 20bp, two BUY votes, sufficient stated confidence, valid current quotes and inventory limits. Severe risk flags veto entry. Consensus cannot manufacture an edge. Residual intervals are logged, not treated as proven win probabilities. Ranked profit uses a planned budget and impact assumption; depth-walk VWAP, partial capacity and future exit liquidity can differ and consume the edge margin. Realized simulated fees/fills remain separate from that estimate.
7. `monitor.py` independently checks held positions on a 30-second target interval using fresh public depth, with no news or LLM calls. It restores entry context from durable audit events and immutable snapshots after restart; guarded position identities prevent a stale exit from closing another position. `storage.py` uses unique fsynced temporary files for atomic, concurrent research/monitor artifacts.
8. `portfolio.py` walks captured depth with configured impact, precision/minimum-size rules and fees. Partial fills and failed exits remain visible. SQLite holds shared cash, positions, fills and chained decisions; verification reconstructs cash and position basis. Idempotent decisions and a process lock prevent duplicate actions after restart. No real order is sent.

Defaults: 20% maximum per position, 40% total exposure, two positions, 0.5% entry risk budget, 1–2.5% ATR stop distance, 2:1 target/stop distance, 3% drawdown halt, and eight-hour timeout. The protective monitor targets 30-second checks independently of the slower 300-second research loop. Network delays, stale quotes, gaps and insufficient depth can postpone checks, prevent liquidation or exceed the stop price. Stopping the service does not liquidate its paper positions.

## Server operation

Examples assume the repository is checked out at `/opt/wallex-trading` with its virtualenv in `/opt/wallex-trading/.venv` (see [DEPLOYMENT.md](../DEPLOYMENT.md)).

The unit template is `deploy/systemd/wallex-paper-agent.service`, rendered and installed by `deploy/systemd/install.sh`. Python dependencies are pinned in `requirements.txt`. It runs as the service user (default `wallex`), loads the secret through systemd credentials, and writes durable state in `/var/lib/wallex-paper-agent`. The 300-second interval is a target between cycle starts; slow requests can stretch cadence. The portfolio and API budget survive restarts.

```bash
sudo systemctl start wallex-paper-agent.service
sudo systemctl status wallex-paper-agent.service --no-pager
sudo journalctl -u wallex-paper-agent.service -n 30 --no-pager
cd /opt/wallex-trading
sudo -u wallex /opt/wallex-trading/.venv/bin/python -m experiments.wallex_agent status --state /var/lib/wallex-paper-agent
```

Equivalent entry point, using the service user and credential environment:

```bash
/opt/wallex-trading/.venv/bin/python -m experiments.wallex_agent serve \
  --model /opt/wallex-trading/artifacts/wallex_agent/quant-universe-v2/model.json \
  --state /var/lib/wallex-paper-agent \
  --artifacts /var/lib/wallex-paper-agent/artifacts \
  --poll-seconds 300 --research-limit 3
```

Do not run a second foreground process against active service state. `run` supports bounded `--cycles 1..12` and optional registered `--symbols`; held positions are monitored even when omitted from that subset. `serve` scans the full manifest. `status` reads the last completed cycle without API calls.

Durable files:

- `status.json`: last complete cycle, source failures, decisions, portfolio and budget.
- `daemon.json`: research process heartbeat, completed cycles, consecutive errors and completion time.
- `protection.json`: latest independent protective check, quote failures, attempted exits and current portfolio.
- `paper-portfolio.sqlite`: shared account and audit ledger.
- `news.sqlite`, `council-cache.sqlite`: evidence and cached research.
- `llm-budget.sqlite`: shared attempts and charged/reserved estimates.
- `artifacts/{cycles,decisions,snapshots}/`: inspectable records and captured inputs.

These paths are relative to `/var/lib/wallex-paper-agent`. A successful status command can show an old cycle: inspect `finished_at` in full status JSON, `last_completed_at`/errors in `daemon.json`, and freshness/errors in `protection.json`. A research cycle can be old while the independent monitor has a newer portfolio; compare their timestamps. Account risk configuration, universe and cutoff are pinned in the database. Changes require an explicit migration or a separately identified account retaining the old evidence. Do not erase a ledger to represent a drawdown halt as a fresh result.

## Provider budget

The secret stays at `/etc/wallex/anthropic.env`, outside Git. The unit exposes a credential file. Overrides: `WALLEX_ANTHROPIC_SECRET_FILE` and `WALLEX_ANTHROPIC_BUDGET_DB`. Never paste the secret into commands, documents or logs.

The shared ledger allows **1,200 reserved attempts** and **$2 charged-or-reserved estimated spend** across restarts. Reservations are atomic; ambiguous POSTs keep their maximum reservation. Budget exhaustion blocks fresh council entries. Existing-position deterministic risk exits do not require LLM responses.

Configured Haiku 5.5 rates are $0.10/input MTok and $0.50/output MTok below 100k tokens, verified against [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) on **2026-10-22**. This is an application accounting ceiling, not a provider-side billing limit; provider billing needs separate reconciliation.

## Frozen quantitative reproduction

Directory: `artifacts/wallex_agent/quant-universe-v2/`.

- Canonical manifest identity: `a78b63415cd2e3777efef7aab94c17f602c5292523ea4149c7e631f8e38c81b9`.
- SHA256 of actual `model.json` bytes: `edfde497f6f13975877e7342d86b385e2c120ce48f68b7cde1fbc529fb302cc9`.
- SHA256 of exact frozen `quant-source.py` bytes: `8bc4521196bed933146a3ce48f36913f990ab2a20d89b8e1cc8889e3999d55b1`.

Canonical JSON and indented file bytes intentionally hash differently. Report/model/protocol pairing uses canonical identities; native tree/source integrity uses byte hashes. The directory also contains hash-verified training configuration, protocol, dependency freeze, runtime versions and reproduction manifest. Large backtest reports are kept out of git.

The training datasets (`data/wallex_agent/universe-v2/...`) are not part of this repository; with them, `quant.train_research` and `quant.simulate_shared_research` in the frozen `quant-source.py` reproduce the model (hashes are checked against `protocol.json`).


Offline tests, with no provider calls:

```bash
cd /opt/wallex-trading
/opt/wallex-trading/.venv/bin/python -m pytest \
  tests/test_wallex_agent_quant.py tests/test_wallex_quant_review.py \
  tests/test_wallex_agent_integration_review.py tests/test_wallex_agent_ranking_review.py -q
```
