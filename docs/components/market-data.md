# Wallex market context and TradingView bridge

This module supplies source-labelled market evidence for the experimental agent. It
has no order, wallet, paid-data, or private TradingView account operations.

## Interface

```python
from experiments.wallex_agent.market import MarketContext, MarketConfig
snapshot = MarketContext(config=MarketConfig()).snapshot("BTCTMN")
```

Research accepts strict uppercase ASCII TMN market symbols that exist in the actual
Wallex public market metadata; malformed or unlisted symbols are rejected. This does
not establish their approval for a challenge account. The execution RiskConfig can
freeze a `research_universe` tuple; an empty tuple permits any structurally valid TMN
symbol whose context market identity matches. `snapshot(symbol, now=None)`
returns JSON-serializable evidence:

| Key | Contract |
| --- | --- |
| `schema_version`, `symbol`, `observed_at` | Version 1; Wallex symbol; UTC Unix cutoff seconds |
| `market` | Existing Market fields: symbol, amount_decimals, minimum_quantity, minimum_notional |
| `market_metadata` | Wallex source, explicit TMN quote, base currency, public venue metadata |
| `candles` | Oldest-first closed hourly `{t,o,h,l,c,v}` bars from Wallex |
| `candle_quality` | Upstream quality, cutoff, selected count, age, cadence validation, partial bars dropped |
| `book` | Sorted positive finite bid/ask price-and-quantity levels, best quotes, spread, REST reception time |
| `technical` | Hourly and UTC-aligned 4-hour summaries using only complete contiguous bars |
| `cross_market` | Optional TradingView Binance BASEUSDT data with explicit status and USDT quote |
| `provenance` | Public route names, SHA256 of canonical primary candles, observed history designation |

The book contains `source_timestamp_available: false` and `source_timestamp: null`:
Wallex REST reception time does not prove freshness of exchange quotes. A caller
supplying `now` gets that deterministic observation cutoff; live use should omit it.
Snapshot persistence belongs to the caller, enabling immutable audit snapshots and
side-effect-free tests. No files are written just by requesting context.

## Data quality and indicators

Defaults request eight days of hourly Wallex data and require at least 72 complete
contiguous bars. The existing public client checks exchange/server clock skew,
actual source cadence and aggregation completeness. Snapshot validation independently
checks finite positive OHLC, finite nonnegative volume, alignment, uniqueness and
contiguity. Incomplete/future candles are dropped before indicator calculations.
Primary history more than 5,400 seconds old relative to its final close fails closed.
The upstream history normalizer may retain a recent contiguous segment after an older
gap; its discarded-bar counts remain in candle_quality.

Indicators: Wilder RSI14 and ATR14; SMA-seeded EMA12/26 and trend; 1/4/24-hour returns;
24-hour population standard deviation of unannualized log returns; last volume divided
by the previous 24-bar mean. Values lacking warmup are null. The 4-hour view uses only
UTC-aligned bins containing all four hourly bars. No price/volume imputation occurs.
These are descriptive features and do not establish a profitable strategy.

## Optional TradingView source

Requested library: [Mathieu2301/TradingView-API](https://github.com/Mathieu2301/TradingView-API).
Inspected source checkout commit: `6a4f8ff9a2b332d0a59dbd27c75201eb84663045`.
The isolated adapter package pins `@mathieuc/tradingview` to `4.0.0`; package-lock.json
pins the resolved npm dependencies and integrity hashes. Install (Node.js 20+):

```sh
cd experiments/wallex_agent/tradingview
npm ci --ignore-scripts --omit=dev --no-audit
```

The bridge was verified with Node.js v24.22.1.
The Python adapter runs a fixed sibling `fetch.mjs` using argument arrays, no shell,
and a minimal environment containing only PATH and LANG. It passes JSON on stdin.
The bridge validates an uppercase Binance USDT symbol, hourly timeframe, maximum 500 bars, timeout and actual
installed package version. Requests use `getCandles` with timeout and AbortSignal;
Python independently imposes a 25-second child deadline (configurable 1..30 seconds)
and validates count, cadence, bounds, closure, freshness, symbol, version and currency.
Child output and stderr are not copied into the agent prompt on errors.

Default 200-bar cross-checks cache within the same UTC hour for at most one hour.
Cache results are deep copies, and no failed request is presented as cached success.
Missing Node, remote rejection, timeout, invalid data or stale data yields explicit
`status: unavailable`; primary Wallex evidence can still be produced. Set
`MarketConfig(tradingview_enabled=False)` to disable the optional request explicitly.

This is an independent unofficial community client. Anonymous feeds can have shorter
history, delays or substitute data; no TradingView session cookies were provided or
used. The inferred target is `BINANCE:<BASE>USDT`, with optional explicit
`MarketConfig(tradingview_symbols={"RENDERTMN": "BINANCE:RNDRUSDT"})` alias mapping.
A missing Binance market degrades explicitly to unavailable. The Binance USDT
cross-check remains USDT throughout. No direct comparison or
conversion to Wallex TMN prices is made. Market data does not supply news content or
execution capability.

## Verification

Offline suite:

```sh
python3 -m unittest tests.test_wallex_agent_market -v
```

The suite covers full-book bounds, missing/crossed/locked/nonfinite quotes, market and
currency isolation, stale/insufficient/gapped/duplicate primary history, partial-bar
exclusion, future perturbation invariance, complete 4-hour bins, indicator conventions,
child failures/timeouts, provenance mismatch, optional-source degradation, cache
immutability/hour rollover, and configuration bounds. It makes no network requests.
