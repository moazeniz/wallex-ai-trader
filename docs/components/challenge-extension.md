# Wallex challenge extension

Status: **research baseline**: public-data backtests and read-only shadow decisions. It never places orders (live execution lives in the arena, see [live-trading.md](live-trading.md)).

## Start here

Run from the repository root with the project virtualenv. The Wallex module itself uses only Python's standard library; the environment also contains the original predictor's pinned dependencies for regression tests.

```bash
PY=.venv/bin/python

# Offline checks; no account credentials, notifications or orders.
$PY -m pytest tests/test_predictor_v0_data.py tests/test_predictor_v0_model.py tests/test_predictor_v0_cli.py tests/test_wallex_challenge.py -q

# Capture a current-market decision. Restarting on the same bar adds no duplicate.
$PY -m experiments.wallex_challenge shadow --symbol BTCTMN \
  --model artifacts/wallex_challenge/btc-hourly-001/model.json \
  --settings config/wallex.research.json \
  --policy config/wallex.account.example.json \
  --output .wallex-state/btc --once
```

Remove `--once` only when intentionally running the foreground observer continuously; it polls every 300 seconds and can be stopped with Ctrl-C. No background service, public port or scheduled job was installed.

For a new research run, use a new empty data/output directory:

```bash
$PY -m experiments.wallex_challenge fetch --symbol BTCTMN --days 30 \
  --settings config/wallex.research.json --output data/wallex_challenge/btc-next
$PY -m experiments.wallex_challenge backtest \
  --dataset data/wallex_challenge/btc-next/dataset.json \
  --settings config/wallex.research.json --output artifacts/wallex_challenge/btc-next
```

Completed output directories are refused. Existing data and run artifacts are retained for provenance. Do not reuse an evaluated period to tune thresholds and then call it an untouched test.

## Architecture and reuse

The existing project supplies signal generation, a Telegram interface, ensemble code and a later chronological predictor experiment. Its recorded `predictor_v0` development result is `INSUFFICIENT_EVIDENCE`. The supplied checkout has no trained ensemble artifacts; parts of the legacy analyzer also contain `/opt/wallex` paths, use external venues and use USD paper balances. Starting `main.py` therefore does not establish a working Wallex challenge engine.

The extension is deliberately an additive experiment with its own entry point; it does not change the earlier predictor's code, settings, data split or reserved holdout. The old 105 offline predictor tests remain passing. The new Wallex data begins after that experiment's reserved period.

`PublicAPI -> validated closed OHLCV -> learned logistic model -> deterministic cost/risk policy -> historical paper broker -> hash-linked decision log -> replay`

The model learns seven past-only OHLCV features with standardized L2 logistic regression. It predicts positive next-open to open-four-bars-later direction. The hourly configuration therefore has a four-hour prediction target. Means, scales, weights and expected returns are learned only from training rows whose labels mature before evaluation. The final 30% of each dataset is evaluated chronologically with no fitting on those rows. This is one small baseline, not the missing GRU/LSTM/XGBoost ensemble and not an LLM agent. Judge acceptance of this AI architecture should be confirmed; its prediction has a real effect on entry/exit proposals.

The cost gate combines learned direction probability and train-only conditional return means. The latter is a rough expectation estimate, not a calibrated confidence bound. An entry requires probability >= 0.60 and estimated gross edge exceeding the assumed round-trip cost. Exits can follow the model, stop, target, timeout or final settlement. Trading rules constrain learned proposals rather than generating artificial model predictions.

The historical broker uses Decimal cash accounting, no leverage or shorts, 20% maximum position notional, 0.5% risk budget, 1% stop, 2% target and a 3% sampled drawdown halt. Stops cannot guarantee these loss limits through price gaps. Entry size is rounded down using observed market precision and limited by prior volume and a simulated current-bar participation cap. Full order-book queue, latency and intra-bar execution paths are not modeled. If stop and target occur within one bar, the stop is applied first. Exit liquidity or minimums can leave residual assets; the result must then report `settled_to_tmn=false`.

Assumptions: 35 bps fee and 10 bps slippage per side, both unverified for the actual subaccount. The minimum order is conservatively 100,000 toman, because public market metadata reports 50,000 while the later organizer reply says 100,000. Neither is silently treated as proof of the credited account's rules.

## API and data checks

Primary sources checked directly from the server:

- [Candles](https://developers.wallex.ir/docs/basic-services-candles)
- [Market metadata](https://developers.wallex.ir/docs/basic-services-markets)
- [Order-book depth](https://developers.wallex.ir/docs/basic-services-depth)
- [Account-specific fees](https://developers.wallex.ir/docs/basic-services-account-fee)
- [Credit subaccounts](https://developers.wallex.ir/docs/trade-credit-intro)
- [Spot order contract](https://developers.wallex.ir/docs/spot-create-order)

Observed public endpoints: `/v1/markets`, `/v1/udf/time`, `/v1/udf/history`, `/v1/depth`. Current documentation also describes `/hector/web/v1/markets`; the starter currently consumes the observed legacy metadata shape, so any schema change fails visibly rather than guessing fields. The old `api-docs.wallex.ir` host failed TLS hostname validation; certificate validation was not bypassed.

A request for resolution 15 returned 60-second candles. Minute-resolution long ranges returned `no_data`, while hourly history worked. The adapter validates actual cadence, pagination boundaries, aligned and unique timestamps, finite OHLCV and complete target intervals. It drops unfinished buckets and retains the newest contiguous segment, without filling missing prices or volume. Research configuration was fixed to hourly data before the first fit; this also matches the existing experiment's four-hour target. Failed source probes are preserved outside the source tree.

Book REST responses lack a source timestamp; local reception time is logged, and freshness of the exchange quote is not claimed. Stale completed candles fail shadow observation. Shadow checkpoints are recovered from a verified audit chain, with a process lock preventing concurrent writes. Source, settings, model and data hashes bind each historical run; replay checks every decision and equity value against stored inputs. A local hash chain detects accidental or partial changes, not an attacker who can replace every artifact and its anchor.
