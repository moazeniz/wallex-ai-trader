# Wallex AI trading system

An AI-led trading engine for the Wallex algorithmic-trading challenge. It reads live Wallex order books and trades,
global prices, the USDT/TMN rate and multi-topic news; Claude decides what to buy and sell and how much; a
deterministic risk layer enforces position limits, stops and the final settlement to TMN. Several strategy variants
run side by side on the same live data ("the arena") so their results can be compared fairly before and during the
challenge. Every decision is written to a tamper-evident log.

## 1. What problem the strategy solves

Measured on Wallex, not assumed:

| Finding | Evidence |
|---|---|
| Wallex TMN prices follow global prices with a delay of minutes | 7 days of 1-minute data, 56 markets: the gap between the Wallex price and the global fair value predicts the next Wallex move (correlation −0.3 to −0.5 at 5–60 min) |
| Small gaps do not pay after costs; large gaps do | Buying at market when the gap is ≤ −1.2% and selling with a limit order at fair value: 53 trades, mean **+0.37 % per trade net of fees and spread**; gaps of −0.4…−0.6 % lose money |
| The cached all-markets endpoint is too stale for trading | `/v1/markets` top-of-book matched live `/v1/depth` only 9 % of the time, so the engine reads live depth |
| Fees | Wallex public schedule (`/v2/structured-fee`): TMN markets 0.30 % taker, 0.25 % maker; USDT markets 0.13 % / 0.10 % |
| The rial matters | USDT/TMN rose from ~195k to ~267k in the 60 days before the challenge (mean +0.5 %/day, daily s.d. 1.6 %); TMN returns of every coin include this component |

The research scripts and datasets behind these numbers are not part of this repository; the order-book recorder
(`services/book_recorder.py`) that collected the live data is.

## 2. Architecture

```
 Wallex WebSocket (order books, trades)   Binance WebSocket (1 s prices)   17 RSS feeds (EN + FA)
   + REST /v1/depth, /v1/trades backup     + REST book tickers backup              │
                  └───────────────────┬──────────────────┘                          │
                              MarketFeed (feed.py, ws_feed.py, binance_ws.py)  NewsFeed (engine.py)
   live books · trade tape · fair value = Binance × USDT/TMN (coin-implied rate while USDT/TMN is closed) · 1-min bars
                                      │
          ┌───────────────────────────┼──────────────────────────────────────────┐
   Rule strategies            Claude as trader / as maker manager            Benchmark
   LagReversion ·             (ai.py, ai_strategy.py): dashboard →            hold BTC
   FairValueMaker             structured decision or market-making plan
          └───────────────────────────┼──────────────────────────────────────────┘
                                      │ actions
                       Engine risk envelope (engine.py), every 2 s
   fast drop guard · position/exposure caps · risk-per-trade sizing · fair-value-aware stops · max hold ·
   drawdown pause/halt · nightly USDT/TMN halt handling · final TMN liquidation
                                      │
     PaperAccount (account.py)  ── or ──  LiveAccount (live_account.py) → LiveExecutor (live.py) → Wallex
     fills walk the live book / trades       only Wallex-reported fills, real fees, balance reconciliation
                                      │
           Watchdog (watchdog.py): if the engine goes silent, exchange-side stop orders protect live positions
```

One process runs every strategy against one shared feed, so all variants see identical data. Order books and
global prices stream in real time (median book age about 0.3 s); the engine decides every 2 seconds, with full REST
snapshots every 8 seconds as a backup. USDT/TMN stops trading every night 23:30-11:30 Tehran time; while it is closed
the fair value uses the rate implied by liquid coin markets (Wallex price / Binance price), calibrated to the last
USDT/TMN price, and USDT itself cannot be traded.

## 3. How the AI decides

Two roles, both through the Anthropic API with schema-validated output:

**Claude as trader** (`AITrader`): every 5-15 minutes, and early when a new opportunity appears, the model receives a
JSON dashboard:

- for each of ~45 eligible TMN markets: mid price, spread, `gap_ask` (how far the best ask is from global fair value
  after the market's normal premium), Wallex and global changes over 15 min / 1 h / 4 h / 12 h / 24 h, 24 h volume
  and liquidity within 0.4 % of the best ask;
- USDT/TMN level, whether that market is open, and its 1 h / 4 h / 24 h / 7 d change (the rial), with measured
  statistics on how rial moves behave;
- 30 recent headlines balanced across crypto, Iranian economy, global macro, forex and geopolitics (marked untrusted);
- its account: equity, cash, open positions with P&L, recent fills, its own notes from the previous cycle;
- in hybrid mode, rule-based signals currently firing (liquid markets only), which it approves or rejects and sizes.

It returns `Decision` (`ai.py`): `market_view`, BUY/SELL actions with size, stop-loss, take-profit, maximum holding
time and a written reason, signal reviews, and notes for its next cycle.

**Claude as maker manager** (`AIMaker`, the configuration intended for the live account): every 15 minutes the model
receives the same dashboard plus its current market-making plan and per-market results, and returns `MakerDecision`:
which markets to quote (up to 8), the quote distance from fair value and the size for each, and which inventories to
sell immediately. The FairValueMaker engine executes that plan every 2 seconds. A guardrail keeps the plan inside the
evidence: markets with a 0.5-3 % spread, margin 0.25-1.0 %, at most 8 % of equity per market.

**Every discretionary entry, exit and size in an AI account comes from the model.** The code only clamps values to
the allowed ranges and enforces the risk rules below (protective exits: stops, time-outs, drawdown, final settlement).
Every input dashboard is saved (gzip) and referenced by its SHA-256 in the decision log next to the model's output.

Variants (`arena-config.json`): Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5; solo, hybrid and maker-manager modes;
"patient" and "active" styles.

## 4. Buy and sell conditions

| Strategy | Buy | Sell |
|---|---|---|
| Claude trader | the model's BUY, with its reason, size, stop and target | model SELL; limit take-profit at its target; stop-loss; max hold; final settlement |
| Claude maker manager | the engine's bid in a market the model chose, at the model's margin and size | the engine's ask, or the model's flatten order |
| LagReversion | best ask ≥ 1.2 % (strict) or 0.8 % (loose) below global fair value after the market's 24 h premium; spread ≤ 1.2 %; 24 h volume ≥ 3 B TMN | resting limit sell at fair value + 0.2 %; stop −4 %; max hold 120 min |
| FairValueMaker | resting bid one tick above the best bid but never above fair × (1 − margin) | resting ask ≥ fair × (1 + margin) and ≥ entry + fees; stop −3 %; max hold 6 h |
| USDC stablecoin maker | the same, on USDC/TMN only (≈2 % spread, price pegged to the dollar) | the same |
| Hold BTC | once at start (benchmark) | final settlement |

Maker protections (`maker_v2`, the AI makers and the stablecoin maker; the original `maker` runs without them as a
control): no new bid in a coin whose global price fell more than 0.4 % in the last 2 minutes, more than 1 % in 4 hours
or more than 3 % in 12 hours; a 20-minute cool-down after buying a market; at most 6 inventories at once; stale
inventory (over 1 hour) is offered at break-even; in the final 3 hours no new bids and asks step down to the best ask.

Resting (maker) orders are filled in the paper account only when real Wallex trades print **through** the order
price, or the opposite best quote crosses it — never on a touch, because queue position is unknown.

## 5. Capital and risk management

Enforced by the engine for every account, whatever the strategy or model proposes:

- position ≤ 35 % of equity (makers: 7-8 % per market); total exposure ≤ 95 % (makers: 45 %); resting buys count
  against cash;
- **risk-based sizing**: position ≤ 1 % of equity ÷ (stop distance + round-trip cost), so a wider stop means a smaller position;
- **stops measured from the best bid at entry**, checked every tick against the live best bid and confirmed on a fresh
  order book before acting (a single snapshot can show a momentary wick);
- **fair-value-aware stops**: when a stop triggers while Wallex has overshot the global price by more than 1 %, the
  engine rests a limit sell just under fair value for up to 3 minutes before falling back to a market sell;
- **fast drop guard**: when a coin's global price falls more than 0.4 % within 60 seconds, resting bids in that market
  are cancelled in every guarded account and new buys are blocked for 2 minutes;
- drawdown from peak ≥ 5 % pauses new entries for 60 minutes; ≥ 10 % liquidates and stops new entries;
- minimum order 100,000 TMN; market orders never take liquidity more than 0.4 % from the best price (live: price-capped
  LIMIT orders, unfilled remainder cancelled);
- only TMN markets outside the organizers' excluded list; spot only, no leverage; never a bid and an ask in the same
  market at once, so no self-trading;
- USDT/TMN is not traded while that market is closed (23:30-11:30 Tehran);
- no new entries in the final hour and **full liquidation to TMN** at the configured end time (`liquidate_at_utc`),
  because the challenge measures realized TMN value (paper arena; the live build below uses a 2-hour / 60-minute window);
- live only: balances reconciled with Wallex every 5 minutes, and the watchdog places exchange-side stop orders if the
  engine stops responding for 2 minutes.

### Live account limits (Liara build)

- **Total exposure:** maker inventory is capped at **35% of equity** (`WALLEX_MAX_EXPOSURE`, default 0.35, inside the
  recommended 30-40% range; the paper test used 45%).
- **Per order and per market:** at most 8% of equity per market (the AI guardrail) and at most 2,000,000 TMN per order.
- **Drawdown:** new entries pause at 5% drawdown; the account halts and liquidates at 10%.
- **End of challenge, settled in TMN:** with `WALLEX_CHALLENGE_END_UTC` set to the official end, no new positions are
  opened from 2 hours before it. From 60 minutes before it, all orders are cancelled and every coin is sold to TMN with
  bounded immediate-or-cancel orders, retried every 30 s per market until the account is flat. Each step is logged as
  a `flatten` record. Positions below the 100,000 TMN minimum order are reported as dust.

### How we use Wallex MCP/API

- **Wallex MCP (official server, `https://api.wallex.ir/mcp`):** the AI's research step and the dashboard page `/market`
  get candles (several resolutions), order-book depth, recent trades and market/ticker data through four public,
  read-only MCP tools. No API key is sent, and every account, order, margin or credit tool is refused before any
  network call.
- **Fallback:** if the MCP server is unavailable, the same data comes from Wallex public REST (marked `source: rest`).
  Every call is logged as `mcp.call`.
- **Wallex public WebSocket + REST:** the engine's live order books and trades (low latency).
- **Wallex private REST:** orders, cancels and balances, and only when live trading is armed. MCP is never used for
  trading.
- Details and the tool list are in [DEPLOYMENT.md](DEPLOYMENT.md#how-we-use-wallex-mcpapi).


## 5a. What the shadow test showed (2026-10-22)

Measured on live Wallex order books with paper accounts, about 6 hours on the first day:

- **Market making around the global price earned steadily in calm conditions**: the first maker completed 4 of 4
  profitable round trips (+0.33 % of the account) before 13:08 UTC.
- **A single coin can jump with no warning.** At 13:08 UTC FIL fell 4.1 % on Binance in one minute while BTC and ETH
  were flat, with no headline in any of our 17 feeds. Every maker holding FIL lost more than its earlier profit, and the
  stops sold into a thin book at the low. This led to the fair-value-aware stop, the fast drop guard and smaller,
  more diversified inventory.
- **Makers lose when they keep buying coins in their own downtrend** (FIL −4 %, PUMP −6 % over 12 hours on a flat-BTC
  day). This led to the downtrend filter.
- **Frequent directional trading loses to fees**, as reported for LLM traders in live competitions; the patient Claude
  Opus trader, which trades rarely, was the only account above zero in the afternoon.
- **Thin markets** (for example IMX, under 1 B TMN a day) produced repeated losing lag trades; signals now require 3 B
  TMN of daily volume.

Each change was deployed as its own commit, and the unchanged accounts keep running as controls, so its effect is
measured rather than assumed.

## 6. Logs, verification and reports

Per account in `/var/lib/wallex-arena/accounts/<name>/`:

- `decisions.jsonl` — every engine start, AI decision (full model output, cost, latency, request id), placed and
  cancelled order, market decision with its signal values, and fill. Each line stores `sha256(previous hash + record)`;
  `DecisionLog.verify` detects any edit, deletion or reordering.
- `account.sqlite` — fills, equity every minute, positions.
- `llm-budget.sqlite` — every model call with token counts and USD cost (daily cap per account).

`python -m experiments.wallex_arena report /var/lib/wallex-arena report.md` builds the shadow-test report
(leaderboard, every trade with its reason, AI statistics, log verification) from these files alone.

## 7. Run it

See the [README quick start](../../README.md#quick-start) for local runs and [DEPLOYMENT.md](../DEPLOYMENT.md) for the
systemd install. On a server it runs as the unit `wallex-arena` (state in `/var/lib/wallex-arena`). Live trading is a
separate, explicit step described in [live-trading.md](live-trading.md); it is off by default (environment switch
plus a `LIVE_DISABLED` lock file).

## 8. Limitations, stated plainly

- The backtest covers 7 days of one market regime; 53 trades is a small sample. The parameter grid was evaluated on
  the same data, so the live shadow test is the real check.
- Paper fills assume our orders do not move the market; with positions of 0.2–1.75 M TMN in markets trading
  10–100+ billion TMN per day this is small but not zero.
- Maker fills ignore queue position at the same price (conservatively, touches never fill).
- LLM decisions are not deterministic; reproducibility comes from saving every input dashboard (gzip, referenced by sha256 in the log), output and
  cost, and from the deterministic risk layer.
