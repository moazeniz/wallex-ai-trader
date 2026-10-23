# Wallex AI Trading Arena — Project Guide

**Team «MeowMeow» · Starcoach Agentic AI Hackathon · Wallex algorithmic-trading challenge · October 2026**

> Persian version (primary): [`PROJECT_GUIDE.fa.md`](PROJECT_GUIDE.fa.md)
>
> **Liara app (production): https://meowmeow-wallex.liara.run** ✅. Checked in a real browser on 2026-10-22 at about 17:50 UTC. It is password-protected (HTTP Basic). It shows:
> - the paper/shadow arena results (snapshot of the server arena taken 17:01 UTC);
> - the shadow-test report, with all 17 hash chains verified;
> - the decision-log viewer;
> - the docs and the 7-day backtest;
> - a `/live` control page. The live engine is **DISARMED**: engine off, no Wallex API key, kill file present.
>
> The page's **AI self-test** made one real model call, answered by `claude-sonnet-5-5` through the team's Anthropic relay in 5.3 s, with `openai/gpt-6.1-sol` on BoxAPI as the automatic fallback route.
>

**Status legend** used throughout this document:

| Mark | Meaning |
|---|---|
| ✅ | **Live**: running on our server now, on live Wallex data |
| 🧪 | **Implemented and tested**: code and offline tests exist, but it is switched off or not yet used in production |
| 🚧 | **In progress**: planned or partly built; do not count it as done |

All numbers below were read from the code, the saved research outputs, and the live arena's own logs on **2026-10-22 at 15:15 UTC**, unless a different time is given. The arena keeps running, so live numbers change.

---

## 1. Elevator pitch

Toman prices on Wallex follow global prices (Binance) with a delay of minutes. We measured this effect and built an engine that trades it in two ways:

- it buys when Wallex is clearly cheaper than the global price converted to toman;
- it quotes limit orders around that global "fair value" to earn the spread at the lower maker fee.

On top of this engine, **Claude makes the discretionary decisions**:

- which markets to trade;
- how wide to quote and how much to commit;
- when to step aside on news, a crash or a rial shock.

A deterministic risk layer that the AI cannot override enforces position caps, risk-based sizing, stops, drawdown halts and the final conversion of everything to toman.

To choose what to run on the real account using evidence instead of hope, we built a **strategy arena**. It runs **17 accounts** side by side on one live feed: rule-based, LLM-led and a benchmark, each with 5,000,000 paper toman. Every decision, input and fill is written to a **hash-chained, tamper-evident log**.

---

## 2. The problem

**The challenge.** Each team trades a credited Wallex account of 5,000,000 TMN:

- spot only, no leverage;
- 100,000 TMN minimum order;
- excluded markets are not allowed;
- everything must be realized in toman at the end.

Judging rewards profit, drawdown control, regular activity and stability. It also rewards a strategy the team can explain, real use of AI ("not just if/else"), risk management, reproducibility, and backtest plus shadow-test evidence.

**What we measured on Wallex itself** (not assumed):

| Finding | Evidence | Where |
|---|---|---|
| Wallex TMN prices lag global prices by minutes | 7 days of 1-minute Wallex and Binance data. The gap between Wallex and global fair value predicts the next Wallex move | `research/v3/scripts/analyze_leadlag.py` (server) |
| Small gaps don't pay after costs; large gaps do | Buy at a gap of −1.2% or more and exit at fair value with a limit sell: **53 trades, mean +0.40% per trade net of fees and spread, 70% winners**. Gaps of −0.4% lose money in every tested setting | `research/v3/data/backtest_tmn.txt`, `hist7d/backtest_lag_TMN.csv` |
| Real fees | Wallex public schedule `/v2/structured-fee`: TMN markets 0.30% taker and 0.25% maker. USDT markets 0.13% and 0.10% | `experiments/wallex_arena/account.py` |
| The cached all-markets endpoint is too stale to trade on | `/v1/markets` best quotes matched live `/v1/depth` only **9%** of the time. The engine therefore reads live depth and the WebSocket | `research/v3/scripts/analyze_live.py` |
| USDT/TMN stops trading every night | 23:30–11:30 Tehran (20:00–08:00 UTC), zero volume, while coin markets keep trading. Fair value then uses the USDT rate implied by liquid coins | `feed.py` (`usdt_halted`, `_update_implied_rate`) |
| The rial is part of every TMN return | USDT/TMN rose from about 195k to about 267k in the 60 days before the challenge | AI system prompt and dashboard context |

---

## 3. What we built, at a glance

| Component | What it does | Status |
|---|---|---|
| **Live market feed** | Wallex WebSocket for books and trades, a Binance WebSocket, REST fallback, fair value, premium, implied USDT rate | ✅ |
| **Strategy arena** | 17 paper accounts on one shared feed: identical data and a fair comparison | ✅ `wallex-arena.service` |
| **Rule strategies** | `LagReversion` (2 variants), `FairValueMaker` (5 variants incl. protected `maker_v2` and USDC-only `maker_stable`), `HoldBenchmark` | ✅ |
| **AI trader** (Claude) | Claude decides BUY/SELL, size, stop, target and hold from a market dashboard. Modes: solo, hybrid and active | ✅ 6 accounts (Haiku, Sonnet, Opus 5.5) |
| **AI-managed market maker** | Claude acts as portfolio manager of the maker engine: it picks markets, margin and size, and can flatten inventory | ✅ 2 accounts (Haiku, Sonnet) |
| **Research paper agent** | Frozen quant models (XGBoost/Ridge), 9 RSS feeds and a three-analyst Haiku council (crypto, macro, skeptic) with a strict consensus gate | ✅ `wallex-paper-agent.service` (baseline) |
| **Deterministic risk engine** | Position and exposure caps, 1% risk sizing, stops, max hold, fast-drop guard, fair-value stop, drawdown pause and halt, final toman settlement | ✅ |
| **Tamper-evident decision logs** | SHA-256 hash chain per account, AI inputs saved as gzip and referenced by hash | ✅ |
| **Live Wallex execution** | Private REST client, live account bridge with fill booking and balance reconciliation | 🧪 25 + 4 tests. **Dry-run account live** ✅. Real orders switched off |
| **Dead-man watchdog** | Places exchange-side STOP_MARKET orders if the engine goes silent | 🧪 installed, disabled |
| **Order-book recorder** | Live depth, trades and Binance quotes every 12 s, for research evidence | ✅ `wallex-book-recorder` |
| **Live dashboard** | Leaderboard, return curves, recent trades with reasons, stream health. HTTPS with a password | ✅ `https://<SERVER_HOST>:9445`. Also on Liara as a snapshot (next row) |
| **Backtests** | 7-day lag backtest, logistic baseline, frozen quant replay | ✅ results saved |
| **Wallex MCP** | Official MCP server discovered and documented (`https://api.wallex.ir/mcp`) | 🚧 not integrated |
| **Liara deployment and dashboard** | Liara app `meowmeow-wallex`. Password-protected dashboard over a hashed snapshot of the arena (results, report, decision logs, docs and backtest), a `/live` ARM/DISARM control page, a boot connectivity self-check, a retry job queue, circuit breakers and `/readyz` | ✅ https://meowmeow-wallex.liara.run. The live engine is DISARMED. `/paper-agent` returns an internal error 🚧 |
| **LLM provider routing** | `LLM_PROVIDER` picks the primary provider, and the other is the fallback on 429/5xx, credit or network errors: Anthropic (`claude-sonnet-5-5`) over plain HTTP through the team relay, or BoxAPI (`openai/gpt-6.1-sol`) with key rotation. Release 1.1.0 of the published repo | ✅ on Liara (primary `anthropic`, 3 BoxAPI keys configured). The Hetzner arena still uses the Anthropic SDK |
| **Real-money go-live** | Run the chosen strategy on the credited account | 🚧 waits for the credit-account API key and the team's go-ahead |

---

## 4. Feature tour

### 4.1 Live market feed ✅

The feed lives in `experiments/wallex_arena/feed.py`, `ws_feed.py` and `binance_ws.py`.

- **Wallex WebSocket (Centrifugo v2).** It subscribes to three channels per market: `buyDepth`, `sellDepth` and `trade`. Wallex allows at most 64 channels per connection, so the feed shards across connections.
  - Live health at 15:06 UTC: **3 connections, 138 channels, 46 markets**, median book age **0.4 s**, about 150,000 messages and **0 parse errors**.
- **REST fallback.** `/v1/depth` and `/v1/trades` run every few ticks. A REST snapshot never overwrites a book that the stream updated more recently.
- **Binance WebSocket** (mini-ticker for 42 pairs) gives global prices about once per second. These feed the fast-drop guard.
- **Fair value and premium:**
  - fair value of `XXXTMN` = Binance `XXXUSDT` mid × Wallex USDT/TMN mid;
  - premium = trailing 24-hour median of ln(Wallex mid / fair).
  - The signal is **gap** = ln(Wallex best ask / fair) − premium.
- **Nightly USDT halt.** The feed detects the halt (no USDTTMN trade for 20 minutes). It then switches fair value to the USDT rate implied by eight liquid coins and blocks USDTTMN orders.
- **Universe.** The 45 highest-volume TMN markets outside the organizers' **345-market exclusion list**, plus USDTTMN.
- **News.** 17 RSS feeds in English and Persian (crypto, Iranian economy, macro, forex, geopolitics), refreshed every 10 minutes. A per-topic quota keeps the mix balanced, and the model receives the headlines labeled **untrusted**.

### 4.2 The strategy arena ✅

One engine process (`engine.py`) runs every strategy against the same feed with a **2-second tick**. On each tick, for every account:

1. Fill resting maker orders that real Wallex trades printed **through**, or that the opposite quote crossed. A touch never counts as a fill.
2. Check drawdown (pause or halt), the final settlement, stops and time-outs.
3. Ask the strategy for actions, then validate and size them against the risk envelope and execute.
4. Record equity every minute and write the leaderboard (`status.json`) every 30 seconds.

**Accounts running now** (`arena-config.json`, 17 accounts):

| Group | Accounts |
|---|---|
| Lag rule | `lag_strict` (1.2% gap), `lag_loose` (0.8% gap) |
| Maker rule | `maker` (control, unprotected), `maker_v2` (protected, smaller and diversified), `maker_wide` (0.6% margin), `maker_tight` (0.3%), `maker_stable` (USDC/TMN only) |
| AI trader | `ai_haiku_solo`, `ai_haiku_hybrid`, `ai_sonnet_hybrid`, `ai_opus_hybrid`, `ai_haiku_active`, `ai_sonnet_active` |
| AI-managed maker | `ai_maker_haiku`, `ai_maker_sonnet` |
| Benchmark | `hold_btc` |
| Live code path | `live_dry_maker`: the real live-order code with a key-less dry-run client. It logs the exact Wallex requests and never sends them |

### 4.3 Rule strategies ✅

- **LagReversion.** Buys at market when the Wallex ask is far below the global fair value. It exits with a resting limit sell at the premium-adjusted fair value, so the exit pays the maker fee and no spread.
- **FairValueMaker.** Rests a bid one tick above the best bid, but never above fair × (1 − margin). After a fill, it rests an ask at or above fair × (1 + margin) and above entry plus both fees.
  - It never has a bid and an ask in the same market at the same time, so **self-trading is impossible**.
- **Maker protections** (in `maker_v2`, `maker_stable` and both AI makers):
  - a cooldown after fills;
  - an inventory cap;
  - pulling bids when the global price falls fast;
  - no new bids in a 4-hour or 12-hour global downtrend;
  - relaxed exits for stale inventory;
  - a 3-hour planned wind-down through limit orders.
- **HoldBenchmark.** Buys BTC once and holds it, as the yardstick.

### 4.4 The AI trader (Claude decides) ✅

`ai.py` and `ai_strategy.py`. Every 5–15 minutes, and early when a new lag signal appears, the account sends Claude a compact **JSON market dashboard** (about 14–16k tokens). It contains:

- for each of about 45 markets: mid, spread, `gap_ask` and `gap_mid`, Wallex and global changes over 15 minutes, 1, 4, 12 and 24 hours, 24-hour volume, and buy liquidity within 0.4%;
- the USDT/TMN level, whether that market is open, and its 1-hour, 4-hour, 24-hour and 7-day change;
- 30 balanced recent headlines (labeled untrusted);
- the account: equity, cash, open positions with P&L, recent fills with reasons, and its own notes from the last cycle;
- in **hybrid** mode, the rule-based lag signals currently firing, which the model must approve or reject and size.

Claude returns a **schema-validated `Decision`** (Pydantic, through structured outputs):

- `market_view`;
- `actions[]` with symbol, BUY or SELL, size fraction, stop-loss, take-profit, max hold, confidence and a written reason;
- `signal_reviews[]`;
- `notes_for_next_cycle`, a short memory carried to the next call.

**Every discretionary entry, exit and size in an AI account comes from the model.** The code only clamps values to the allowed ranges and enforces the risk rules. The only automatic exits in an AI account are protective: stop, timeout, drawdown halt and final settlement.

Variants:

- models: Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5;
- modes: `solo` (dashboard only) and `hybrid` (also reviews rule signals);
- styles: `patient` and `active` (looks for 3–8 well-reasoned swing trades a day).

### 4.5 The AI-managed market maker ✅

In `AIMaker`, Claude is the **portfolio manager** of the maker engine:

- every 15 minutes it returns a `MakerDecision`: up to 8 markets, a margin and a size for each, a reason, and a `flatten[]` list for immediate exits on risk;
- the deterministic engine executes the plan on every tick.

The model also receives `maker_stats_last_6h` (realized P&L per market and how long inventory has been stuck), so it learns from its own results. A **guardrail** keeps it inside the region where the engine has earned:

- markets with a spread outside 0.5–3% are dropped;
- the margin is clamped to 0.25–1.0%;
- the size is capped at 8% of equity per market.

Dropped markets are logged as `maker_plan_guardrail`.

### 4.6 Research paper agent: quant plus analyst council ✅ (baseline)

`experiments/wallex_agent/` runs as `wallex-paper-agent.service` with its own 5M paper account. It was built first and kept as the conservative baseline.

**How it decides:**

- frozen per-market XGBoost and Ridge forecasts of the 8-hour return, with calibration-only arm selection and cash as an allowed choice;
- 9 RSS feeds;
- a **three-analyst Claude Haiku council** (`crypto_analyst`, `macro_analyst`, `skeptical_risk_analyst`) that runs in parallel and must cite news IDs;
- a risk gate that needs a quantitative edge above costs plus 20 basis points **and** at least two BUY votes. Severe risk flags veto the trade.

**Live record:**

- since 08:27 UTC: **79 cycles and 157 decisions, all HOLD**;
- reasons: not among the top net opportunities 109, no buy consensus 30, edge below costs 30, manipulation flag 15, spread too wide 10;
- the decision chain is verified, and accounting is reconciled;
- LLM use: 166 attempts and $0.32 of its $2 cap.

This baseline shows careful AI reasoning, but it **does not trade**. That is why the arena was built (section 9.3).

### 4.7 Risk management ✅

See section 7 for the full table. Highlights:

- **Risk-based sizing.** A position is at most 1% of equity ÷ (stop distance + 0.7% round-trip cost).
- **Stops are measured from the entry bid.** A bug we found live (the spread alone triggered a stop) led to this fix.
- **Fair-value stop.** When Wallex has overshot the global price by more than 1%, the stop rests a limit sell just under fair value for up to 3 minutes, then falls back to a market sell.
- **Fast-drop guard.** If a coin's global price falls more than 0.4% within 60 seconds, every protected account cancels its bids there and blocks buys for 2 minutes. This fired **97 times** today.
- **Drawdown limits.** A 5% drawdown pauses new entries for 60 minutes. A 10% drawdown liquidates the account and halts it.
- **End of challenge.** No new entries in the final hour, then **full liquidation to TMN** at `liquidate_at_utc`, which is 2026-10-23 19:00 UTC (22:30 Tehran).

### 4.8 Realistic paper execution ✅

`account.py`:

- **Market orders walk the live depth** and never use liquidity more than 0.4% from the best price for buys (1% for exits). No liquidity is invented.
- **Maker fills count only when real trades print through our price**, or the opposite best quote crosses it. Queue position is unknown, so a touch never fills.
- Real fee schedule, the 100,000 TMN minimum on buys **and sells**, and equity marked at the best bid (the liquidation value).

### 4.9 Live Wallex execution 🧪 (with a dry-run account ✅ live)

`live.py`, `live_account.py` and [`components/live-trading.md`](components/live-trading.md):

- **Private REST client** with these calls:
  - create order: `POST /v1/account/orders`;
  - cancel: `DELETE /v1/account/orders/{client_id}`;
  - order detail and open orders;
  - balances, credit sub-account balances and account fee.
  - It sends the `sub-account-client-id` header for credit accounts and never logs the API key (`x-api-key: ***`).
- **"Market" orders are bounded LIMIT orders.** A buy is priced at most at its bound, quantity is rounded down to the step size, and the client re-reads the order after 1.5 s and cancels any remainder.
- **Safety:**
  - a create-order POST is **never retried**. An ambiguous result is settled by looking the order up by `client_id`, or tracked as `unknown`;
  - order creation is throttled to 20 per 10 s;
  - every order has its own `ARENA_<ms>_<hex>` client id;
  - the order state survives restarts.
- **Hard switches**, checked before any network call:
  - no orders while the file `/var/lib/wallex-arena/LIVE_DISABLED` exists (**it exists now**);
  - no orders unless `WALLEX_LIVE_ENABLED=1`;
  - only TMN markets that are not excluded;
  - at least 100,000 TMN and at most `max_order_tmn` (2M by default) per order.
- **LiveAccount bridge.** The same strategies trade the real account through the same engine. It books **only fills that Wallex reports**, at the real fee, and reconciles with exchange balances every 5 minutes.
- **Dry run, live ✅.** `live_dry_maker` has been producing exact Wallex order requests on live data since 12:17 UTC: **5,212 logged entries**, with the key always redacted.

### 4.10 Dead-man watchdog 🧪

`watchdog.py` (systemd unit `wallex-live-watchdog`, installed but disabled) is a separate small process.

- If the engine's `status.json` goes stale for 2 minutes, it acts once:
  1. it cancels our resting orders;
  2. it places a real **STOP_MARKET** sell on Wallex under every coin we hold, 3% below the bid.
- From then on the exchange protects the positions even if our server dies.
- When the engine is back, the watchdog removes its emergency stops.
- It obeys the same two live switches. 4 tests.

### 4.11 Tamper-evident decision logs ✅

Each account writes `decisions.jsonl`. Every line holds `prev`, `hash = sha256(prev + canonical record)` and `record`. `DecisionLog.verify` detects any edit, deletion or reordering.

- **All 17 account chains verified** today, about 36,000 records in total.
- Every AI input dashboard is saved as gzip under `ai_inputs/`, and the log references its SHA-256.
- Every model call is logged with its model, request id, latency, USD cost and full parsed output.

A real example is in section 10.

### 4.12 Research tooling ✅

All on the server, under `/opt/wallex/research/v3/`:

- **Order-book recorder** (`wallex-book-recorder`, every 12 s). It runs 28 TMN and 14 USDT markets plus Binance. So far: **83,533 book snapshots, 50,625 Binance quotes and 19,621 trades**.
- **History fetcher, lead-lag analysis, executable-price analysis, lag backtest and P&L breakdown.**
- **Event watcher** that reports fills, errors, risk events, failed AI calls and services going down.

A copy of the recorder and the dashboard code is in this repo under `services/`. 🚧 The backtest and analysis scripts are still on the server only, not yet in the repo.

### 4.13 Live dashboard ✅

`https://<SERVER_HOST>:9445` (HTTPS with a valid certificate, Basic Auth user `meowmeow`; the password is kept on the server only). It refreshes every 30 seconds and shows:

- engine and stream health;
- the USDT/TMN rate and whether that market is open;
- the guarded markets;
- the leaderboard (approach, model, return, max drawdown, trades, fees, AI cost);
- return curves;
- recent trades with the reason the strategy or AI gave.

✅ A password-protected Liara dashboard serves the same views over a hashed snapshot of the arena at https://meowmeow-wallex.liara.run, plus the `/live` control page. See the header note.

---

## 5. Architecture

### 5.1 Components

```mermaid
flowchart LR
  subgraph Sources["Data sources (public, read-only)"]
    WWS["Wallex WebSocket<br/>books + trades"]
    WREST["Wallex REST<br/>/v1/depth /v1/trades /v1/markets"]
    BWS["Binance WebSocket<br/>+ REST book tickers"]
    RSS["17 RSS feeds<br/>EN + FA"]
  end
  subgraph Engine["wallex-arena (one process, 2 s tick)"]
    FEED["MarketFeed<br/>fair value, premium, gap,<br/>1-min bars, USDT halt"]
    NEWS["NewsFeed<br/>topic quotas"]
    RULES["Rule strategies<br/>LagReversion, FairValueMaker,<br/>HoldBenchmark"]
    AIT["AITrader<br/>solo / hybrid / active"]
    AIM["AIMaker<br/>Claude manages the maker"]
    RISK["Risk envelope<br/>caps, sizing, stops, guards,<br/>drawdown, final settlement"]
    PAPER["PaperAccount x16<br/>depth-walk fills, real fees"]
    LIVEACC["LiveAccount (dry run)<br/>LiveExecutor"]
  end
  CLAUDE["Anthropic API<br/>Claude Haiku / Sonnet / Opus 5.5"]
  STORE[("Per account: decisions.jsonl (hash chain),<br/>account.sqlite, ai_inputs/*.json.gz,<br/>llm-budget.sqlite, status.json")]
  WALLEXPRIV["Wallex private API<br/>(switched off)"]
  DASH["Dashboard (HTTPS :9445)"]
  WD["Dead-man watchdog<br/>(switched off)"]

  WWS --> FEED
  WREST --> FEED
  BWS --> FEED
  RSS --> NEWS
  FEED --> RULES
  FEED --> AIT
  FEED --> AIM
  NEWS --> AIT
  NEWS --> AIM
  AIT <--> CLAUDE
  AIM <--> CLAUDE
  RULES --> RISK
  AIT --> RISK
  AIM --> RISK
  RISK --> PAPER
  RISK --> LIVEACC
  LIVEACC -.-> WALLEXPRIV
  PAPER --> STORE
  LIVEACC --> STORE
  STORE --> DASH
  STORE --> WD
  WD -.-> WALLEXPRIV
```

### 5.2 One engine tick (every 2 s)

```mermaid
sequenceDiagram
  participant F as MarketFeed
  participant E as Engine
  participant S as Strategy (rule or AI)
  participant A as Account
  participant L as DecisionLog
  E->>F: poll (WebSocket is live, REST every few ticks)
  E->>E: fast-drop guard (Binance -0.4% in 60 s: pull bids)
  loop each account
    E->>A: fill resting orders that real trades printed through
    E->>A: drawdown check (5% pause / 10% halt)
    E->>A: stops (confirmed on a fresh book), max hold, final settlement
    E->>S: decide(ctx)
    S-->>E: actions (market_buy / place / cancel / market_sell)
    E->>E: allowed market? entries allowed? guarded? size = min(caps)
    E->>A: execute (depth walk or resting order)
    A->>L: append hash-chained record
  end
  E->>E: equity every 60 s, status.json every 30 s
```

### 5.3 One AI decision (asynchronous, never blocks the tick)

```mermaid
sequenceDiagram
  participant S as AITrader / AIMaker
  participant D as Dashboard builder
  participant C as Claude (structured output)
  participant B as Budget ledger
  participant E as Engine and risk layer
  S->>D: build JSON (markets, rial, news [untrusted], account, notes, signals)
  D-->>S: dashboard (SHA-256, saved as gzip)
  S->>B: under the daily USD cap?
  S->>C: system prompt (cached) + dashboard, schema = Decision / MakerDecision
  C-->>S: parsed decision (view, actions or plan, reasons, notes)
  S->>B: record tokens and USD
  S->>S: clamp ranges, tradable symbols only, one entry per market
  S->>E: actions or maker plan (guardrail drops out-of-band markets)
  E->>E: risk caps, guards, minimums, then execution and log
```

### 5.4 Data and state

| Path (server) | Content |
|---|---|
| `/var/lib/wallex-arena/status.json` | Leaderboard, stream health, guarded markets, USDT halt state |
| `/var/lib/wallex-arena/engine.jsonl` | Hash-chained engine events (seeding, guard triggers) |
| `/var/lib/wallex-arena/accounts/<name>/decisions.jsonl` | Hash-chained decision log |
| `…/account.sqlite` | Fills, positions, equity every minute |
| `…/ai_inputs/*.json.gz` | The exact dashboard sent to the model for each call |
| `…/llm-budget.sqlite` | Every model call: tokens and USD |
| `…/live.jsonl`, `live_orders.json` | (Live or dry-run) exact requests and order tracking |
| `/var/lib/wallex-paper-agent/` | Research agent: portfolio, audit chain, council cache, budget |
| `/opt/wallex/research/v3/data/` | Recorder SQLite, 7-day history, backtest outputs |

---

## 6. AI design

### 6.1 Models and cost (measured)

| Use | Model | Cadence | Measured |
|---|---|---|---|
| AI trader (patient hybrid) | `claude-opus-5-5` | 15 min | 43 calls, mean 9.2 s, about $0.08 per call |
| AI trader (hybrid, active) | `claude-sonnet-5-5` | 5–10 min | 137 calls, 5.6–5.7 s |
| AI trader (solo, hybrid, active) | `claude-haiku-5-5` | 5 min | 252 calls, 4.1–5.7 s, about $0.002 per call |
| AI maker manager | Haiku / Sonnet 5.5 | 15 min | 64 calls, 9–11 s |
| Analyst council (paper agent) | `claude-haiku-5-5` | per cycle | 166 attempts, $0.32 total |

**Arena total:** **500 model calls, $10.04**. **All 496 AI decisions in the logs at analysis time were valid and schema-conforming** (0 API errors, 0 parse failures). A daily USD cap applies per account (`daily_usd`).

Other details:

- The system prompt is sent with `cache_control: ephemeral`, so it is cached.
- Effort is set to `medium`, and the output is parsed with `messages.parse(output_format=…)`.

**Providers (release 1.1.0, the Liara image).** `ClaudeTrader` routes through two plain-HTTP clients: `anthropic_http.py` (direct key or the team relay) and `boxapi.py` (OpenAI-compatible, round-robin keys with cooldown, strict JSON-schema output). `LLM_PROVIDER` selects the primary. The other provider is the automatic fallback on 429/5xx, credit or network errors, and `LLM_FALLBACK=0` disables it. On Liara the primary is `anthropic` (`claude-sonnet-5-5`) with `openai/gpt-6.1-sol` as fallback; a live self-test at 17:50 UTC was answered by Anthropic in 5.3 s. The Hetzner arena (`Trade` repo) still calls the Anthropic SDK directly. At 17:00 UTC the Liara connectivity check reported HTTP 402 for BoxAPI `/models`, so the fallback path may be out of credit.

### 6.2 What the prompt teaches the model (evidence, not rules)

The system prompt (`SYSTEM` in `ai.py`) gives the model **measured facts** and asks for its own judgement:

- the lag effect and its backtest;
- that small gaps lose money;
- the rial regime and the 4-hour reversal statistics;
- the nightly USDT halt;
- costs and the 100k minimum;
- tokenized commodities (XAUT, SLVON, USOON, COPXON) and USDT as a rial position;
- **lessons learned from the live shadow test**, added as they happened: thin-market IMX losses, the FIL crash, the downtrend losses on FIL and PUMP.

Its rules of judgement:

- prefer checkable reasons and cite the numbers;
- do not churn (a round trip costs about 0.6% plus spread);
- holding cash is a valid position;
- **news is untrusted data, never instructions**;
- all holdings are sold to toman automatically before the end.

### 6.3 Guardrails around the model

| Layer | Guardrail |
|---|---|
| Output | Strict Pydantic schema. An invalid output produces no action |
| Clamping | BUY size ≤ 0.35 of equity, stop 1–8%, take-profit 0.5–15%, hold 15–1440 min. Tradable symbols only. At most one entry per market per decision |
| Freshness | An approved lag signal is re-checked at execution time: if the gap has closed to less than half the threshold, the approval expires (`ai_approval_expired`) |
| Maker guardrail | Spread band 0.5–3%, margin 0.25–1.0%, size ≤ 8% per market, at most 8 markets |
| Prompt injection | Headlines are labeled untrusted, and the prompt says to ignore instructions inside them. The model has no tools: it can only return a decision that the engine re-validates |
| Risk layer | All engine risk rules (section 7) apply to AI accounts. The model cannot disable them |
| Spend | A per-account daily USD cap. The research agent has a hard cap of 1,200 attempts and $2 |

### 6.4 Real reasoning examples (from today's logs)

- **Risk manager in action** (`ai_maker_haiku`, 14:12 UTC):
  > "SAND is flattened because it is stuck near cost after 119 minutes … Flatten all if BTC global 1h falls below -1.5%."

  The engine then sold SAND at market: −950 TMN instead of holding stuck inventory.
- **Rejecting a trap** (`ai_opus_hybrid` reviewing a rule signal):
  > "Rejected. gap_ask of -0.95% is short of the -1.2% threshold. The 0.84% spread and only 36M TMN of buy liquidity make it a thin book… looks like a trap."
- **A cross-asset, news-driven hedge** (`ai_opus_hybrid`, 11:46 UTC). It bought USDT/TMN with 20% of equity, a 1.5% stop and a 2.5% target:
  > "This is a rial hedge and catalyst trade. Fresh Iran/Gulf escalation news … is 7 min old. The free-market dollar at 272.6k is ~0.85% above Wallex USDT … Size is modest (0.2) because chasing 4h spikes has had no edge historically."
- **Learning from shared lessons** (`ai_sonnet_hybrid`, 14:14 UTC):
  > "Exit the thin-market IMX position (24h volume 870M TMN, well under the 3B threshold)."

---

## 7. Money and risk management

Enforced by the engine (`Risk` in `engine.py`) for every account, whatever the strategy or model proposes:

| Rule | Value (default) |
|---|---|
| Max position | 35% of equity |
| Max total exposure | 95% of equity; resting buy orders count against cash and exposure |
| Risk per trade | Position ≤ 1% of equity ÷ (stop distance + 0.7% round-trip cost): a wider stop means a smaller position |
| Stop-loss | On every position. Checked every tick against the live best bid and **confirmed on a fresh order book** before selling. Measured from the entry bid |
| Fair-value stop | If the bid is more than 1% below the global anchor, a limit sell at fair − 0.5% for 180 s, then a market sell |
| Max holding time | On every position (strategy- or AI-set) |
| Fast-drop guard | Global −0.4% in 60 s: cancel bids and block buys in that market for 120 s |
| Drawdown pause | 5% from peak: no new entries for 60 minutes |
| Drawdown halt | 10% from peak: liquidate and stop new entries |
| Market impact | Market buys use liquidity within 0.4% of the best ask only |
| Minimum order | 100,000 TMN (the organizers' value, stricter than the API's 50,000), on buys and sells |
| Allowed markets | TMN spot only; 345 organizer-excluded markets blocked in code; no leverage, no shorting |
| USDT/TMN | No orders while the market is closed at night |
| End of challenge | No new entries in the final hour; **full liquidation to TMN** at 2026-10-23 19:00 UTC |
| Live only | Each order between 100k and 2M TMN; kill-file and environment switches; watchdog with exchange-side stops |

Strategy-level money management:

| Strategy | Sizing and limits |
|---|---|
| `maker` | 15% of equity per bid, up to 5 markets, at most 60% exposure |
| `maker_v2` and the AI makers | 7–8% per market, up to 6 inventories, 45% exposure, 20-minute cooldown |
| `lag_*` | 30% per trade, capped by liquidity within 0.4%, at most 4 positions |
| AI trader | The model chooses up to 35%. The engine's 1%-risk sizing usually binds first |

**Research agent** (separate account):

- 20% per position, 40% total exposure, at most 2 positions;
- 0.5% risk per trade, ATR stops of 1–2.5%, 8-hour timeout;
- a **3% drawdown halt**.

---

## 8. Buy and sell conditions

| Strategy | Buy | Sell |
|---|---|---|
| **LagReversion** (`lag_strict` 1.2%, `lag_loose` 0.8%) | Market buy when `gap_ask` < −threshold, spread ≤ 1.2%, book fresher than 15 s, 24-hour volume ≥ 3B TMN. Biggest discount first, at most 4 positions | Resting limit sell at fair × e^premium × (1 + 0.2%), repriced as fair moves; a market sell if the bid already reaches it. Stop −4%; max hold 120 minutes; final settlement |
| **FairValueMaker** (`maker`, `_wide`, `_tight`) | Resting bid = min(best bid + 1 tick, anchor × (1 − margin)) in the busiest markets with spread 0.6–3%. Margin: 0.4%, 0.6% or 0.3% | Resting ask = max(best ask − 1 tick, anchor × (1 + margin), entry × (1 + 2 × 0.25% + 0.1%)). Stop −3%; max hold 6 hours |
| **Protected maker** (`maker_v2`, `maker_stable`, AI makers) | As above, but **no new bid** in these cases: during the final 3-hour wind-down; at the inventory cap; global −0.4% in 2 minutes; global trend < −1% over 4 hours or < −3% over 12 hours; within 20 minutes of the last fill in that market | Stale inventory (more than 60 minutes) or wind-down: ask at max(best ask − tick, anchor, break-even after fees). In the final hour, just inside the spread, accepting up to −0.3% |
| **AI trader** | The model's BUY with its reason, size, stop and target. In hybrid mode, also an approved lag signal (exit at fair value, stop −3%, 120 minutes) | Model SELL (all or part); resting take-profit at its target; stop; max hold; drawdown halt; final settlement |
| **AI maker** | The engine's maker bids in the markets, margins and sizes the model planned (after the guardrail) | Maker asks as above; the model's `flatten` sells at market immediately |
| **HoldBenchmark** | 98% of equity in BTC once at start | Final settlement |
| **Research agent** | Quant edge > costs + 20 bp, at least 2 of 3 analysts vote BUY with enough confidence, valid quotes, within limits | ATR stop, 2:1 target, 8-hour timeout, 2 of 3 SELL votes or a severe risk flag, 3% drawdown halt |

---

## 9. Backtest and shadow-test evidence

### 9.1 Backtest: Wallex lags global prices ✅

**Setup** (`backtest_lag.py`):

- 7 days of 1-minute Wallex and Binance candles, 2026-10-15 to 10-22, for the eligible TMN markets;
- realistic fills: buy at the next minute's open plus half the live spread; the limit exit fills only if a later high is **strictly above** it; the stop is taken at the low minus half the spread;
- real fees.

A grid of 32 parameter sets:

| Threshold | Edge | Stop | Hold | Trades | Sum of returns | Positive markets |
|---|---|---|---|---|---|---|
| **1.2%** | 0.2% | 4% | 120 min | **53** | **+21.3%** (mean **+0.40%** per trade, **70% win**) | 8 of 10 |
| 1.2% | 0.2% | 2% | 120 min | 54 | +18.6% | 8 of 10 |
| 0.8% | 0.2% | 4% | 120 min | 215 | +14.9% | 8 of 14 |
| 0.6% | 0.2% | 4% | 120 min | 428 | −22.2% | 5 of 14 |
| 0.4% | 0.2% | 4% | 120 min | 968 | −206% | 0 of 15 |

At the best setting, the best markets were PUMP (+1.66% per trade over 6 trades), SAND (21 trades, 76% wins) and ARB (11 trades).

> Note: an earlier run of the same backtest, quoted in the model's system prompt and in [`components/arena.md`](components/arena.md), gave +0.37% per trade. The saved output on the server gives +0.40%.

### 9.2 Earlier baselines (kept, reported honestly)

| Experiment | Result |
|---|---|
| `wallex_challenge`: logistic regression, 7 OHLCV features, BTC and ETH hourly, chronological 70/30 split | **0 trades** in 213 evaluation hours on each market. The cost gate never cleared. Buy-and-hold: BTC +1.93%, ETH −0.41% |
| `wallex_agent` V2 frozen quant replay, shared account across 16 markets | **−8.73%**, 32 trades, 10.08% max drawdown. Every calibration winner lost out of sample |

These negative results are why we moved from predicting candles to the **measurable lag and spread edge**.

### 9.3 Shadow test: the arena on live Wallex books ✅

The arena has run since 2026-10-22 09:21 UTC on live order books, with real fees and maker fills only on trades that print through our price.

Snapshot at **15:15 UTC** (about 6 hours; a calm, falling market; BTC benchmark −0.53%):

| Account | Approach | Return | Max DD | Closed trades (wins) | AI cost |
|---|---|---|---|---|---|
| ai_opus_hybrid | AI trader, Opus | **+0.011%** | 0.21% | 1 (0) + USDT hedge open | $3.37 |
| lag_strict | Lag 1.2% | 0.000% | 0 | 0 | – |
| ai_haiku_solo / hybrid | AI trader, Haiku | 0.000% | 0 | 0 (chose cash) | $0.14 / $0.15 |
| ai_sonnet_hybrid | AI trader, Sonnet | −0.098% | 0.10% | 1 (0) | $1.95 |
| ai_sonnet_active | AI trader, Sonnet, active | −0.167% | 0.19% | 1 (0) | $2.77 |
| ai_maker_sonnet | AI-managed maker | −0.252% | 0.26% | 2 (1) | $1.42 |
| ai_maker_haiku | AI-managed maker | −0.308% | 0.31% | 2 (0) | $0.09 |
| lag_loose | Lag 0.8% | −0.433% | 0.97% | 1 (0) | – |
| hold_btc | Benchmark | −0.534% | 0.81% | – | – |
| ai_haiku_active | AI trader, Haiku, active | −0.724% | 0.76% | 4 (0) | $0.16 |
| maker_wide | Maker 0.6% | −0.761% | 1.11% | 3 (2) | – |
| maker_v2 | Protected maker | −0.809% | 0.94% | 6 (3) | – |
| maker_tight | Maker 0.3% | −1.326% | 1.60% | 4 (3) | – |
| maker | Maker (control) | −1.427% | 1.75% | 8 (5) | – |

The newest accounts (`maker_stable` from 15:06 and `live_dry_maker`, which is dry-run only) had no closed trades yet.

**Update: snapshot at 17:01 UTC** (about 7.7 hours; this is the snapshot served on Liara and in the shadow-test report):

- **Overall:** 1 of 17 accounts is in profit after fees. 15 of 38 closed trades made money. AI spend was $12.36 in total.
- **Best account:** `ai_opus_hybrid` at +0.018% (max drawdown 0.211%, 50 of 50 AI calls valid).
- **AI traders and AI makers:** between 0.000% and −0.732%, with max drawdowns from 0.098% to 0.762%.
- **Benchmark:** `hold_btc` at −0.745% (max drawdown 1.067%).
- **Rule makers:** between −0.845% and −1.793% (max drawdowns 1.11%–2.37%). A falling ARB move at 16:59 UTC triggered the protected stop for three makers.
- **Not traded yet:** `lag_strict`, `maker_stable`, `ai_haiku_solo`, `ai_haiku_hybrid` and `live_dry_maker` were still flat.
- **Integrity:** all 17 hash chains verified.

**What the shadow test taught us**, each lesson turned into a tested change:

1. **The maker's normal trading earns.** Before 13:08 UTC the plain maker completed **every round trip at a profit** and peaked at **+0.33%** (equity 5,016,596). `maker_tight` peaked at +0.24%.
2. **One idiosyncratic crash erased it.** FIL fell 4.1% globally within a minute at 13:08–13:09 UTC. It filled every maker bid, and stops sold into a thin book at 256k while fair value was about 261k. Each maker lost 13k–43k TMN.
   - **Changes:** the fair-value stop, the Binance stream with a 2-second tick and the fast-drop guard (commits `fa41f8c`, `bbfa232`), and a smaller, diversified inventory (`9e32bca`).
3. **Coins in a steady downtrend keep filling a long-only maker** (FIL −4% and PUMP −6% over 12 hours).
   - **Change:** the 4-hour and 12-hour downtrend filter (`30d0200`).
4. **Thin markets gap and stay gapped.** IMX lost for four accounts.
   - **Change:** a 3B TMN volume floor for lag rules and AI lag reviews (`14cb424`, `6369616`), and the lesson added to the AI's evidence (`f348515`).
5. **Stops measured from the ask fired on the spread.**
   - **Change:** stops are measured from the entry bid (`4f3ba93`).
6. **Careful AI management lost less in the crash.** The AI makers had chosen smaller sizes (−0.25% and −0.31% versus −0.8% to −1.4% for the rule makers). The patient Opus trader is the only account in profit.

**Honest reading:** after six hours and 33 closed trades in a falling market, **no strategy has proven itself profitable yet**. The maker-family accounts closed 21 trades with 13 winners. The AI traders closed 7 with none winning, but they mostly stayed in cash. The protections were added after the crash, and the overnight data will show whether they help.

---

## 10. Decision log format and a real example

Each line of `decisions.jsonl` has the form `{"prev": <hash of previous line>, "hash": sha256(prev + canonical JSON of record), "record": {...}}`.

| `record.type` | When |
|---|---|
| `engine_start` | Process start, with the full strategy parameters and risk settings |
| `ai_decision` | Every model call: model, input SHA-256 and file, parsed decision, USD, latency, request id |
| `maker_plan`, `maker_plan_guardrail` | The AI maker's applied plan, and the markets the guardrail removed |
| `place`, `cancel` | Resting orders with the signal that justified them |
| `decision` | Market buy or sell with reason, signal values and result |
| `fill` | Quantity, price, notional, fee and rate, maker or taker, realized P&L, cash after, reason |
| `skipped`, `rejected`, `ai_action_rejected`, `ai_approval_expired` | Why an intent was not executed |
| `risk_pause`, `risk_halt`, `guard_pull`, `stop_not_confirmed` | Risk events |
| `live_dry_run`, `live_refused` | Live code path: the exact request (key redacted) or the local refusal |

**Real chain from `ai_maker_haiku`** (14:12:59 UTC, shortened):

```json
{"prev": "0a639301…", "hash": "b8376352…", "record": {
  "type": "ai_decision", "model": "claude-haiku-5-5", "ok": true,
  "latency_s": 13.0, "usd": 0.0031, "request_id": "req_011CgHQtudaXyiG4fNc51Jhe",
  "input": {"file": "ai_maker_haiku/ai_inputs/1792678363-75233f55c4f3.json.gz",
            "sha256": "75233f55c4f3f95060a27ad1fa4678cece1722819eef917993d8b73d9910ba9c"},
  "decision": {
    "market_view": "Global crypto is flat to mildly up (BTC global 1h +0.27%) … SAND is flattened because it is stuck near cost after 119 minutes.",
    "plan": [{"symbol": "AVAXTMN", "margin_pct": 0.5, "size_fraction": 0.06,
              "reason": "Spread 1.14% inside guardrail, buy_liq 185m, global 1h now +0.58% …"},
             {"symbol": "APETMN", "margin_pct": 0.6, "size_fraction": 0.04,
              "reason": "Spread 0.63%, 24h -3.33% downtrend, so wider margin …"}],
    "flatten": ["SANDTMN"],
    "notes_for_next_cycle": "… Flatten all if BTC global 1h falls below -1.5%."}}}
{"prev": "621acc96…", "hash": "77af7b55…", "record": {
  "type": "fill", "symbol": "SANDTMN", "side": "SELL", "price": 14684.0,
  "qty": 17.0544, "notional_tmn": 250426.36, "fee_tmn": 751.28, "fee_rate": 0.003,
  "liquidity": "taker", "pnl_tmn": -949.92, "reason": "ai flatten (risk)",
  "cash_after_tmn": 4985411.64}}
{"prev": "77af7b55…", "hash": "5bec95af…", "record": {
  "type": "decision", "op": "market_sell", "symbol": "SANDTMN",
  "reason": "ai flatten (risk)", "result": "filled"}}
```

A judge can open the referenced `ai_inputs/…json.gz`, check its SHA-256, and see exactly what the model saw.

---

## 11. How to re-run and evaluate

Python 3.12+ (the server runs 3.14). The arena needs `anthropic`, `pydantic` and `websockets`; versions are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt

# 1. Offline test suite (no network, no keys)
pytest -q                     # 425 collected: 422 passed, 3 skipped (about 5 minutes on the server)

# 2. Run the arena (paper), with your own key file (chmod 600, line ANTHROPIC_API_KEY=...)
python -m experiments.wallex_arena run experiments/wallex_arena/arena-config.json ./arena-state ./anthropic.env

# 3. Watch and evaluate
python -m experiments.wallex_arena status ./arena-state           # leaderboard: equity, return, max DD, fills, wins, fees, LLM $
python -m experiments.wallex_arena report ./arena-state report.md # Markdown report from the logs (see the known issue below)
python -c "from experiments.wallex_arena.account import DecisionLog as D; from pathlib import Path; print(D.verify(Path('arena-state/accounts/maker/decisions.jsonl')))"

# 4. Order-book recorder (evidence) and dashboard export
python services/book_recorder.py ./books.sqlite 12
```

The repository also has `make` shortcuts for the same steps: `make install-dev`, `make test`, `make test-arena`, `make run-arena ANTHROPIC_ENV=…`, `make arena-status`, `make arena-report`, `make run-recorder`, and the emergency `make kill-switch`. Unit templates are in `deploy/systemd/` and installed with `install.sh`. 🧪 These were added while the repository was being restructured, and this guide has not verified them on the server.

On the server, the same commands run as systemd units:

| Unit | Role |
|---|---|
| `wallex-arena` | The arena engine. State in `/var/lib/wallex-arena`, key passed as a systemd credential |
| `wallex-book-recorder` | Order-book recorder |
| `wallex-dashboard-export.timer` and `wallex-dashboard-web` | Dashboard data export and the HTTPS page |
| `wallex-paper-agent` | Research baseline |
| `wallex-live-watchdog` | Dead-man watchdog (disabled) |

**Research agent:**

```bash
python -m experiments.wallex_agent status --state /var/lib/wallex-paper-agent
```

Its frozen quant replay can be reproduced byte-for-byte from `artifacts/wallex_agent/quant-universe-v2/`. The manifest, model and source SHA-256 values are in [`components/paper-agent.md`](components/paper-agent.md).

**Backtest** (scripts on the server, under `research/v3/scripts/`; 🚧 being moved into the repo):

```bash
python fetch_history.py data/hist7d 7
python backtest_lag.py data/hist7d data/books.sqlite TMN
```

**Fixed ✅:** `report` used to raise `KeyError: 'actions'` on AI-maker accounts. It now counts `plan` entries too. The shadow-test report served on Liara (17:01 UTC snapshot) was generated by this command and covers all 17 accounts.

---

## 12. Mapping to the judging criteria

### 12.1 Trading results (scored on the credited account)

| Criterion | How we address it | Status |
|---|---|---|
| **Profit %** (realized in TMN) | The strategy with the measured edge (lag and spread). The final settlement to TMN is automatic. The live system will run the AI-managed maker chosen by the arena | Engine ✅. Live run 🚧 (needs the account key and the team's go-ahead) |
| **Drawdown control** (30) | 1% risk sizing, stops, fair-value stop, fast-drop guard, downtrend filter, 5% pause and 10% halt, watchdog. Max drawdown of every AI account in the shadow test is ≤ 0.31% | ✅ paper; watchdog 🧪 |
| **Activity** (15) | The maker places and re-quotes around the clock (about 900–3,200 order placements per maker account so far). AI managers re-plan every 15 minutes | ✅ paper |
| **Stability** (15) | No retried POSTs, lookup by client id, throttling, reconciliation, a state that survives restarts (26 restarts today, all resumed). **0 tick or account errors** in the logs, 0 crashes | ✅ paper; live 🧪 |

### 12.2 Technical review

| Criterion | Evidence | Status |
|---|---|---|
| **Strategy logic** (40) | A measured lag effect, fair value and premium, exact buy and sell rules (section 8), with a backtest and live lessons behind every rule | ✅ |
| **Real use of AI** (35) | Claude makes every discretionary decision in the AI accounts: entries, exits, sizes, market choice, quote width, risk exits. It reads market, rial and news data and writes its reasons. Rule-only control accounts run beside it. A second AI system (the analyst council) gates the research agent | ✅ |
| **Risk and capital management** (30) | Section 7: engine-enforced and AI-independent | ✅ |
| **Reproducibility** (25) | One config, one command, 433 offline tests in the published repo, saved AI inputs with hashes, hash-chained logs, a frozen and hashed quant artifact, and a SHA-256 MANIFEST of every snapshot file. GitHub repo `moazeniz/wallex-ai-trader` with CI (lint and tests, platform kit, secret scan, dependency audit, Liara preflight, container image scan), green on `133b5f8` | ✅ code and CI. 🚧 the repo was private when checked (17:55 UTC), and no git tag yet |
| **Backtest and shadow test** (20) | A 7-day lag backtest with a 32-set grid, two earlier baselines, and a 17-account live shadow test with every trade logged | ✅ |

### 12.3 Required documentation content

| Required | Where in this guide |
|---|---|
| Architecture and core logic | Sections 4, 5 and 8 |
| How AI and Wallex tools (API/MCP) are used | Sections 4.4–4.6, 6 and 13 |
| Buy and sell conditions | Section 8 |
| Money and risk management | Section 7 |
| How to re-run and evaluate | Section 11 |
| Decision log for every trade | Section 10, plus `decisions.jsonl` for every account |

---

## 13. How we use Wallex tools

| Wallex tool | Use | Status |
|---|---|---|
| WebSocket `wss://api.wallex.ir/connection/websocket` | Real-time books and trades for 46 markets | ✅ |
| `GET /v1/depth`, `/v1/trades` | Seed and fallback; fresh-book confirmation before a stop sells | ✅ |
| `GET /v1/markets` | Metadata (tick and step sizes, minimums, 24-hour volume), universe ranking | ✅ |
| `GET /v1/udf/history` | Candles for bar seeding and research | ✅ |
| `GET /v2/structured-fee` | Fee schedule (used in the paper accounts) | ✅ |
| `POST/DELETE/GET /v1/account/orders…`, `/v1/account/openOrders`, `/v1/account/balances`, `/v1/account/balances-detail`, `/v1/account/fee` | Live execution, reconciliation and fee check | 🧪 tested with a fake transport; dry-run requests logged live |
| Official **Wallex MCP** (`https://api.wallex.ir/mcp`, Streamable HTTP, about 52 tools, about 5 requests per rate window) | Planned for the AI manager's account reads (balances, fees) | 🚧 documented in `components/live-trading.md`, not yet integrated |

---

## 14. Reliability, security and hardening

**Engine:**

- Errors are isolated per tick and per account and logged; they never stop the loop.
- Systemd restarts the engine on failure.
- All state (cash, positions, resting orders, AI plans and notes, live order tracking) lives on disk, so a restart resumes where it left off.

**Feed:**

- The WebSocket shards around the 64-channel limit, reconnects, and replies to server pings.
- A REST snapshot never overwrites a newer streamed book.
- A book's timestamp is the older of its two sides.
- Stale books (older than 15–30 s) are ignored for decisions.

**Stops:** a stop is confirmed against a fresh REST book before selling (`stop_not_confirmed` is logged when a wick does not confirm).

**Live safety:** two independent switches (an environment variable and a kill file), a market whitelist, minimum and maximum order size, no retried POSTs, an ambiguous-order lookup, throttling, and the dead-man watchdog.

**Secrets:**

- Keys live only in mode-600 files under `/opt/wallex/private/` and are passed to services as systemd credentials.
- The API key is redacted from logs and `repr`.
- `.gitignore` excludes `*.env`, `private/` and password files.

**Sandboxing:** the services run as the unprivileged `wallex-paper` user with `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, and memory and CPU caps. The dashboard server runs as a `DynamicUser` with no capabilities and read-only files, behind HTTPS and Basic Auth.

**AI safety:** schema-only output, no tools, untrusted news, clamped ranges, a per-account budget, and the deterministic risk layer.

---

## 15. Deployment

| Where | What | Status |
|---|---|---|
| Hetzner server `<SERVER_HOST>`, `/opt/wallex/Trade` (git history, latest commit `12daf5f`) | Arena, recorder, dashboard, research agent as systemd services | ✅ |
| Liara app `meowmeow-wallex` (Docker, `/data` disk, non-root, arena checks run at build time) | Password-protected dashboard over the arena snapshot, `/live` control page (DISARMED), AI self-test, `/healthz` and `/readyz`. Deployed through CI/CD with automatic rollback (`deploy.yml`) | ✅ https://meowmeow-wallex.liara.run (the live engine stays off) |
| Credited Wallex account | `ai_maker_sonnet`-style AI-managed maker, starting at 30–40% of capital, with the watchdog enabled | 🚧 needs the credit-account API key and sub-account id in the server's private env file, plus the team's go-ahead |

---

## 16. Testing evidence

Full offline suite on the server (2026-10-22): **425 tests collected, 422 passed, 3 skipped, 13 subtests passed, in 4 min 47 s**. No network or keys are needed.

Other test evidence:
- **Published repo (release 1.1.0):** 433 tests collected. GitHub CI run on `133b5f8` is green across 6 jobs: lint and tests (Python 3.14), platform kit, gitleaks secret scan, pip-audit, Liara preflight, and the container image (build, smoke test, non-root, SBOM, CVE scan).
- **Liara image:** the arena test files (including the BoxAPI, live, watchdog and WebSocket tests) run at image build time.

| Area | Test files | Tests |
|---|---|---|
| Arena engine, strategies, account, protections | `test_wallex_arena.py` | 18 |
| Live REST client (fake transport) | `test_wallex_arena_live.py` | 25 |
| Live account bridge | `test_wallex_arena_live_account.py` | 4 |
| WebSocket feed | `test_wallex_arena_ws.py` | 10 |
| Dead-man watchdog | `test_wallex_arena_watchdog.py` | 4 |
| Research agent (LLM, news, market, quant, execution, monitor, daemon, service, universe, reviews) | `test_wallex_agent_*.py`, `test_wallex_quant_review.py` | 209 |
| Logistic baseline and shadow observer | `test_wallex_challenge.py` | 30 |
| Original predictor and exchange regression tests | `test_predictor_v0_*.py`, `test_exchange.py` | 125 |

**Live evidence** (2026-10-22, 09:21–15:15 UTC):

- 17 accounts; all hash chains verified;
- 500 valid AI decisions ($10.04);
- 76 fills;
- more than 14,500 resting orders placed;
- 46 guard pulls in accounts;
- 0 engine errors.

---

## Liara platform rules we discovered and handled

The Wallex dashboard and its disarmed engine run on Liara, so these rules apply to it directly: deploy limits, rollback, env merging, and the Iranian runtime, which blocks Binance. The rest were found while building Negahban, our Liara SRE agent. All three of our projects share this knowledge. Each rule below was found from an API response, console text or our own experiment, checked on the team account on 2026-10-22/23, and handled in code.

| # | Liara rule or limit | How we found it | How our system handles it |
|---|---|---|---|
| 1 | **New apps need one week of balance.** `POST /v1/projects` returns **429** *"You need enough balance to cover one week's cost"* unless the team balance covers 168 h of the team's current run-rate plus the new app. | API response. Our first replica scale-out on 2026-10-23 was refused this way (Negahban `evidence/scaling-attempt1-blocked-by-balance-rule`). | Negahban's setup wizard warns about it before proposing a new app. The meow-lb replica controller checks 168 h × run-rate **before** it creates a replica, and refuses the scale-out cleanly instead of failing half-way. |
| 2 | **Basic-bundle limits.** 20 deploys per app per 24 h (it behaves like a rolling window), 128 MB source upload, 5-minute build timeout. Builds occasionally stall at *fetching source* and end `TIMEDOUT` with no build log. | Bundle metadata in `/v1/me` (`maxDeploymentCountInDay`, `maxSourceSize`, `buildTimeout`), plus stalls seen in our own deploys. | Our deployers poll each release to a final state. A build that times out never moves traffic: the previous release keeps serving, and Bazargard's deployer checks that it still does. The Wallex deployer cancels a build stuck past the limit (`POST /v2/releases/{id}/cancel`). Then we simply redeploy. Negahban's cards show deploy-quota use and add cooldowns. The replica controller counts the quota per replica app. Source archives exclude data (`.liaraignore`). demo-shop uses the standard bundle (50 deploys a day). |
| 3 | **No zero-downtime on the basic bundle.** A resize or redeploy causes about **20 s of HTTP 502**. | Measured during resize experiments. | Approval cards state "≈20 s of 502" before you approve. Negahban's probe counts 502 as a failure and doesn't hide it. Autopilot uses a 10-minute cooldown and hysteresis. For apps that need it, Negahban suggests the standard bundle with zero-downtime. |
| 4 | **Rollback isn't in the public OpenAPI.** The console calls `POST /v1/releases/{id}/rollback`. | We compared the OpenAPI spec with the console's own requests. | We use that endpoint: Negahban's rollback card, the Bazargard and Wallex release scripts, and the CI deploy job's automatic rollback after a failed smoke test. Negahban also archives every source it deploys, so it can redeploy a known-good source if the endpoint fails. |
| 5 | **Container exec refuses Iranian networks**, including Liara's own runtime. The websocket answers *connection failed*. | Experiment: refused from Arvan and from inside Liara, works from Hetzner (Finland). | Exec relay outside Iran: Liara → Arvan → SSH tunnel → Hetzner. The Liara token is added outside Iran. The terminal runs allow-listed read-only commands directly, needs an approval for other commands, and blocks destructive ones. If the relay is down, the exec tools say so and everything else keeps working. |
| 6 | **Virtual servers (IaaS) can't be deleted for 24 h after creation.** The API answers **409**; the rule itself appears only as console text. | API 409 during our IaaS end-to-end test, plus the console wording. | The create card shows the hourly cost **and the 24 h minimum cost**. A delete before `deletable_from` is refused with the time it becomes possible. Root passwords from VM details are stripped before they reach the UI or the LLM. |
| 7 | **`healthCheck` format is undocumented.** `command` must be an array, durations are in seconds, and `startPeriod` must be ≤ 3000. | Release-API validation errors during template deploys. | Templates generate this format. A template deploy with a health check reached READY (Negahban evidence `e2e-deploy_template-final-*.json`). |
| 8 | **A brand-new `*.liara.run` address takes about 3 minutes to start routing.** | The first live run of the setup wizard timed out while verifying. | Verification of a newly created app now waits up to 6 minutes before it reports failure. |
| 9 | **No native replicas or horizontal autoscaling.** `actions/scale` accepts only 0 or 1; only vertical plans exist. | API and console. There is no replica count and no load-balancer product for apps. | Negahban autopilot scales vertically within an owner-set plan range and daily budget. We also built **meow-lb** (https://meow-lb.liara.run), our own load balancer with a deterministic replica controller in which each replica is a separate app. In the live test on 2026-10-23 at 15 requests/s it made 2 scale-outs and 2 scale-ins. |
| 10 | **No webhooks or alerts API** for crashes, OOM, failed deploys or resource pressure. | The OpenAPI spec and the console have none. | Negahban polls every 60 s (probe, metrics, logs, events, releases). It also offers a push-ingest endpoint with a per-app token for our own apps, and owner-editable alert rules. |
| 11 | **`update-envs` replaces the whole env list and restarts the app.** | Observed while changing env vars: the list we send becomes the whole env list, and the app restarts. | Merge, then update: our code reads the current list, changes one key and sends the full list (Negahban, Bazargard's `infra/liara_apply.py`, Wallex's `deploy_liara.py`). The card warns about the restart. Negahban keeps a private snapshot so a deleted secret can be restored without anyone retyping it. |
| 12 | **Metrics and logs API quirks.** Metrics take `since` as an **absolute unix timestamp**. Logs work only through **v2**; v1 answers *"upgrade your client"*. Logs come back at most 100 lines per call. | API responses during development. | The client always sends absolute timestamps and uses v2 logs. It pages backwards and re-reads the boundary second with de-duplication, which fixed lines being dropped at page boundaries. During a deploy, CPU and memory are summed per timestamp across the old and new containers. |
| 13 | **The runtime is inside Iran.** Large downloads from abroad stall, and many foreign hosts are filtered: Binance and KuCoin are unreachable. Anthropic has to go through our gate; it doesn't serve Iran, and we keep its key outside the Iranian runtime. | Experiments: the catalogue download from Finland stalled; the exchange probes in Wallex's boot self-check failed. | Bazargard seeds its 430 MB catalogue from a domestic Arvan mirror (bearer token + SHA-256 check). Wallex probes Binance, MEXC and KuCoin at boot and uses the first one that answers (MEXC from Liara). All three apps reach Anthropic through the Arvan LLM gate and relay, with BoxAPI, which is reachable from Iran, as the automatic fallback. |
| 14 | **Email alerts need a paid mail plan.** The only plan, `m2`, costs **1,475,000 toman** a month and needs a domain. | The mail-plan list in the API. | We didn't buy it. Negahban shows alerts with an in-app bell and browser notifications, and has Bale and webhook adapters ready (disabled until configured). |
| 15 | **Prices are toman per hour.** The monthly `displayPrice` is price × 720, and an app's cost is plan + bundle, where the bundle price depends on the plan. | `/v1/me` plan tables and `/v1/usage-report`. | Every cost card computes the hourly and monthly delta as plan + bundle. Negahban shows the balance and the hours left at the current run-rate. The replica controller enforces a daily budget in toman. |

---

## 17. Limitations and honest notes

- **No proven profit yet.** The shadow test covers about 6 hours in one falling market. The backtest covers 7 days, and its parameters were chosen on the same data, so it is in-sample. 53 trades is a small sample.
- **Real orders have not been sent yet.** The live path has run in dry-run mode only. Queue position, real fill rates, the credit account's actual fee and how Wallex handles our order flow are all unverified.
- **Paper fills assume our orders don't move the market.** Positions are 0.25–1.75M TMN in markets that trade billions per day.
- **The long-only maker carries crash risk**: a 4% idiosyncratic drop costs several round trips of profit. The protections were added after the crash and are not proven yet.
- **LLM outputs are not deterministic.** Reproducibility comes from saving every input, output and cost, and from the deterministic risk layer.
- **Known bugs and gaps:**
  - one recorded `maker_v2` USDC sale (about 92.9k TMN, +770 TMN) was below the 100k minimum. It happened before the minimum was enforced on sells, and it was left in the history unedited;
  - the system prompt still quotes +0.37% per trade, while the saved backtest output says +0.40%.
- **Not yet done:**
  - Wallex MCP integration;
  - running the engine itself on Liara (it is deployed DISARMED; the arena runs on the server);
  - a public GitHub repo with a tagged release;
  - moving the research scripts into the repo;
  - fixing the `/paper-agent` page on Liara.
- **The BoxAPI fallback may be out of credit:** at 17:00 UTC the Liara connectivity check got HTTP 402 from BoxAPI `/models`.
- **The older explainer** [`components/arena.fa.md`](components/arena.fa.md) mentions a 10-minute AI cadence and 396 tests. The current values are 15 minutes for AI makers and 425 tests.

---

## 18. Roadmap

1. **Go live** with the AI-managed maker at 30–40% of capital, with the watchdog on. Read the actual fee from the first fills. Scale up if real fills match paper. 🚧
2. **Overnight evidence**: protected versus unprotected makers through the USDT halt and any further sharp moves. Then one documented, tagged promotion of the best variant. 🚧
3. **Wallex MCP** for the AI's account reads (balances, fee, open orders). 🚧
4. ~~BoxAPI provider, Liara deployment and Liara dashboard~~: done in release 1.1.0 ✅. Still open: GPT-driven AI-maker accounts in the arena for a side-by-side comparison. 🚧
5. **Maker improvements from the research**: inventory skew, a volatility-based spread floor, and leaning quotes with the lag signal. 🚧
6. ~~Fix `report`~~ ✅. Move the research scripts into the repo, make the GitHub repo public and tag the traded code. 🚧

---

## 19. Team

**Team «MeowMeow»**, Starcoach Agentic AI Hackathon, October 2026.
