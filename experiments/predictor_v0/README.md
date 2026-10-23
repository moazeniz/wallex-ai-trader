# Predictor v0 — final review fix pass / dev-003

`dev-003` is `COMPLETED`; the fixed development screen is `INSUFFICIENT_EVIDENCE`. Tree selected 27 non-overlapping trades (BTC 4, ETH 23), below 100 pooled and 25 per asset. Its pooled delayed mean is also negative. These are historical `reference_price_proxy` results and remain paper-only research.

Source/test fixes: local commit `b724ce10cde450af8fe33b671cb4908beb084e0d`. No config, estimator, feature, target, cost, threshold, schedule or date tuning occurred. No legacy source, M1.8, holdout, Kaggle, OpenRouter, LLM, deployment or push was used. `dev-001` and `dev-002` retain all 20 original file hashes. Native model and development prediction bytes match dev-002.

## Commands and observed results

Run from `C:/Users/Administrator/Downloads/code/Trade` using the existing isolated environment. A completed run ID is refused; use the next unused ID for any authorized rerun.

```powershell
experiments/predictor_v0/.venv/Scripts/python.exe -m pytest tests/test_predictor_v0_data.py tests/test_predictor_v0_model.py tests/test_predictor_v0_cli.py -q --tb=short
experiments/predictor_v0/.venv/Scripts/python.exe -m experiments.predictor_v0.run fetch --config experiments/predictor_v0/config.json
experiments/predictor_v0/.venv/Scripts/python.exe -m experiments.predictor_v0.run develop --config experiments/predictor_v0/config.json --run-id dev-003
experiments/predictor_v0/.venv/Scripts/python.exe -m experiments.predictor_v0.run forecast --config experiments/predictor_v0/config.json --run-dir artifacts/predictor_v0/dev-003 --symbol BTCUSDT
experiments/predictor_v0/.venv/Scripts/python.exe -m experiments.predictor_v0.run forecast --config experiments/predictor_v0/config.json --run-dir artifacts/predictor_v0/dev-003 --symbol BTCUSDT --candles data/predictor_v0/candles.csv.gz --as-of 2026-01-01T00:00:00Z
```

- Offline suite: exit 0, **105 passed in 7.07s**.
- Fetch: exit 0, 109.17397510000228s, 74/74 cached archives checksum-verified.
- Develop: exit 0, 37.39198690000194s wall / 38.203125s CPU for loading through evaluation and metrics persistence, excluding final report/manifest writes; created 2026-09-06T21:11:35.055993Z.
- Live forecast: native process exit **2**, `SOURCE_UNAVAILABLE: unable to retrieve public Binance candles`; 2.641957399999228s. No live forecast was produced.
- Offline replay: exit 0 in 1.558038499999384s; `PAPER_ONLY`, `research_only=true`, `replay=true`, `NO_TRADE / EDGE_BELOW_MARGIN`; gross 0.00009060706361196935, net assumption -0.0029051695947029943.
- Completed-ID refusal: `RUN_ALREADY_COMPLETED`, native exit 2, dev-003 manifest unchanged.

Application failures exit 2; valid abstention/replay exits 0. Some PowerShell wrappers report 1 for a nonzero child; subprocess capture and explicit `exit $LASTEXITCODE` verified the native status. `--as-of` requires `--candles`; current-time live suggestions still expire five minutes after cutoff. Forecasting requires the completed run's config/code/model/schema/metadata hashes and fixed four-hour contract. Older artifacts require their matching code.

## Data, causality and execution limits

Default interval: [2022-12-01T00:00:00Z, 2026-01-01T00:00:00Z). 54,094 normalized rows; each asset has 27,047 / 27,048 expected hourly timestamps (coverage 0.9999630286897367). Each is missing 2023-03-24T13:00:00Z. No gap was filled, no conflicting duplicate accepted; aggregate duplicate count is 0.

The two recorded raw-close anomalies at 2023-03-24T12:00:00Z are BTC 12:39:41.646Z (delta -1218353ms) and ETH 12:39:43.061Z (delta -1216938ms), against 12:59:59.999Z. Each entry now carries symbol, row index 564, archive URL, period and source type. Raw close is accepted only inside [open, open+1h); every deviation is recorded and normalized exclusive close is open+1h. Monthly/daily period and UTC hour-grid checks run before requested-range clipping. Units: 50 monthly archives in ms, 24 in us. The normalized candle hash is unchanged.

Train: 43,658 purged labeled rows. Development: **8,824 scored rows, 4,412 per asset**, including the two terminal rows without a complete delayed reference. The old 8,822 metric count was wrong. Both terminal rows are off schedule; base/delay comparisons retain all 2,206 common valid opportunities with zero opportunity exclusions. Feature warmup/gap exclusions: 196 rows. Each feature uses completed past candles only; base labels use entry open at cutoff and exit open four hours later, strictly before the block boundary.

Python 3.11.0; NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, XGBoost 3.2.0, pytest 9.1.1. No package was added or upgraded. Existing scikit-learn dependency threadpoolctl 3.6.0 scopes embedded calls and inspects native limits. CLI environment caps are set before numerical imports. Fitted/reloaded XGBoost and all three loaded BLAS/OpenMP pools report 4 threads; CPU only, seed 42, fixed TREE_SETTINGS.

## Forecast and policy evidence

Returns are fractions: 0.01 = 1%; 1 bp = 0.0001. MAE uses return units, MSE squared-return units. Only tree/Ridge/zero have forecast metrics. Cash/trend/buy-and-hold have no fabricated forecasts.

| Arm / scope | n | MAE | MSE | OOS R2 vs zero |
| --- | --- | --- | --- | --- |
| tree / pooled | 8824 | 0.007612895116624546 | 0.0001376156064240276 | 0.00021566277353202157 |
| tree / BTCUSDT | 4412 | 0.00560952225197327 | 0.0000706704433028158 | -0.001955203994128407 |
| tree / ETHUSDT | 4412 | 0.009616267981275819 | 0.0002045607695452394 | 0.000963456308940347 |
| ridge / pooled | 8824 | 0.007647292529854371 | 0.00013904440244021343 | -0.010164612510600524 |
| ridge / BTCUSDT | 4412 | 0.005634017007426223 | 0.00007148559579780597 | -0.013512316787443357 |
| ridge / ETHUSDT | 4412 | 0.009660568052282518 | 0.00020660320908262085 | -0.009011436436426434 |
| zero / pooled | 8824 | 0.007609700999391534 | 0.00013764529138933217 | 0 |
| zero / BTCUSDT | 4412 | 0.005607574668562161 | 0.00007053253780318702 | 0 |
| zero / ETHUSDT | 4412 | 0.009611827330220908 | 0.0002047580449754773 | 0 |

Tree MSE is below both baselines pooled and for ETH. BTC MSE beats Ridge but is worse than zero. Tree MAE beats Ridge but is worse than zero in all scopes.

| Policy | Opportunities | Selected | Mean net | Gross cumulative | Net cumulative | Ending USD | Sampled drawdown | Capital-time fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tree | 2206 | 27 | 0.004358545997428325 | 0.1052439207021163 | 0.05980821068207409 | 10598.08210682074 | -0.01456734006748095 | 0.012214745003852093 |
| ridge | 2206 | 4 | -0.0013786007613152684 | 0.0031395072659146095 | -0.0028620867386743853 | 9971.379132613256 | -0.007349226568423073 | 0.0018138155393740596 |
| zero | 2206 | 0 | null | null | null | 10000 | null | 0 |
| cash | 2206 | 0 | null | null | null | 10000 | null | 0 |
| trend | 2206 | 887 | -0.0029993903822430287 | -0.023143177658996872 | -0.7418102417293799 | 2581.8975827062004 | -0.7543880327263475 | 0.41271013906805004 |
| buy_and_hold | 2 | 2 | 0.0032064332003424734 | 0.006220572201375507 | 0.003206433200342529 | 10032.064332003425 | 0 | 1 |

Two equal $5,000 sleeves reinvest independently. Net cumulative return is (BTC ending equity + ETH ending equity)/$10,000 - 1. Costs are 10 bps fee and 5 bps slippage per side; LONG requires predicted net strictly greater than 0.001 at 00/04/08/12/16/20 UTC. Cash/zero have null trade-only statistics and unchanged $10,000 equity.

Buy-and-hold uses the common contiguous span **2025-07-01T00:00:00Z through 2025-12-31T20:00:00Z** (4,412 hours), one entry/exit per sleeve with the same costs. It is continuously invested; scheduled policies hold cash and pay costs on each selected block. BTC passive net is -0.18541800447609325; ETH is 0.1918308708767782. Its endpoint-only sampled drawdown 0.0 omits all intervening losses and cannot be interpreted as low risk. All scheduled policies have the same contiguous span here; no gap or opportunity is dropped for this comparison.

Tree gross/base/doubled/delayed cumulative returns are 0.1052439207021163 / 0.05980821068207409 / 0.017064759900562443 / -0.00197819964978585. Delayed mean is -0.00007413648234626617; the delayed cumulative correction replaces dev-002's incorrect pooled compounding. Ridge delayed cumulative is 0.008411169718120393. All stresses retain selections and use independent sleeves.

Tree capital-time exposure is 543302.3930162982 invested USD-hours / 44479225.136911176 available USD-hours = 0.012214745003852093. Its 108 selected asset-hours and 0.003059836808703536 selected-row fraction are separate quantities. Capital is marked at realized exits; the denominator includes idle/cash time.

| Tree entry month | Trades | Fraction of trades | Net P&L USD |
| --- | --- | --- | --- |
| 2025-07 | 2 | 0.07407407407407407 | -42.13200522936859 |
| 2025-08 | 1 | 0.037037037037037035 | -64.24413699676643 |
| 2025-09 | 1 | 0.037037037037037035 | 60.97411359022681 |
| 2025-10 | 9 | 0.3333333333333333 | 126.89745183050309 |
| 2025-11 | 9 | 0.3333333333333333 | 108.01765090122262 |
| 2025-12 | 5 | 0.18518518518518517 | 408.569032724924 |

ETH supplies 23/27 selections and $498.5228893205631 of the $598.0821068207406 tree P&L. October and November each contain one third of trades; December contributes $408.569032724924. The small, concentrated sample is not evidence of statistical independence or executable profitability.

## Fixed screening rule

The overall label and every criterion are persisted in **metrics.json and generated report.md**.

| Criterion | Result | Observed | Required |
| --- | --- | --- | --- |
| base_mean_net_positive_per_asset | PASS | {"BTCUSDT":0.004990669797583586,"ETHUSDT":0.004248611423488279} | > 0 each |
| delayed_mean_net_positive_pooled | FAIL | -0.00007413648234626617 | > 0 |
| doubled_mean_net_positive_per_asset | PASS | {"BTCUSDT":0.001980210466727722,"ETHUSDT":0.0012403749357128918} | > 0 each |
| full_declared_development | PASS | true | true |
| per_asset_selected_count | FAIL | {"BTCUSDT":4,"ETHUSDT":23} | >= 25 each |
| pooled_selected_count | FAIL | 27 | >= 100 |
| tree_mse_below_zero_and_ridge | PASS | {"ridge":0.00013904440244021343,"tree":0.0001376156064240276,"zero":0.00013764529138933217} | tree < both |

Precedence: fixtures/reduced dates → PIPELINE_ONLY; full development with fewer than 100 trades or fewer than 25 in either asset → INSUFFICIENT_EVIDENCE; sufficient counts plus all edge criteria → CANDIDATE_FOR_FORWARD_TEST; sufficient counts failing any edge criterion → NO_EDGE. Invalid data/tests mean ERROR, not a successful screening label. The screen is heuristic; no retuning was performed.

## Artifacts and integrity

Run artifacts are under `artifacts/predictor_v0/dev-003/`. **The data manifest is shared at `data/predictor_v0/data-manifest.json`, outside the run directory**, alongside candles and raw archives. Prior shared files are preserved as `pre-final-fix-data-manifest.json` and `pre-final-fix-candles.csv.gz`. Fetch errors persist separately as `fetch-failure-<time_ns>.json` and preserve/restore the successful manifest/candle pair.

| Artifact | SHA-256 |
| --- | --- |
| Input config | `cd31c35f6d62c7ae4d719e14f31e8dfb2dfeffe6419f2edd37b98916403d0aa8` |
| Predictor Python code | `cb72e22ddd84551e207f755e9a2147a8172156bfb54d90a6c0c7fef97fa9c765` |
| Normalized candles | `e96bf707921be18e41f315871cbce6f5d3daba8b803517a98c53678830bfdc5e` |
| Shared data manifest | `9cbc6fce4b8daf2c1f34ea19a0d3557ddc388a8609bb5d4cf671fd55dfd6186d` |
| agent-log.md | `9788c53a118ea69b95e90be54d8f7f8e80b026a6ab4aab0c5080cef6dc9e2dd4` |
| config.json | `b2e8cf6c7c03478672625582dcca0ae68b2190a265b8d8f9de7e5e95dd968f9d` |
| development-opportunities.csv.gz | `0e556ffe3ea2823aeaf00d59d0f474bc6f9a402321ffe335a5e877c6e34ecc8e` |
| development-predictions.csv.gz | `770e23f83c653565995a6122ec66810c8db7570f4b1f8b40c39ea407013f173f` |
| feature-schema.json | `68a59bcefe16e88069459373e6348b78b80548e548c8183970b90c2587e52973` |
| metrics.json | `1ac4c001c9874175471bf64697cb7033cb0ff81be71455502df566d50c27de9a` |
| model-metadata.json | `c4e9d4aa8157227fc18ad5dffd204b40211941359fc0e767c340ea0caa2f198b` |
| model.json | `7943e86fdc048a6dda432d255cec6a58e76bb56edb78e148e6bcbf68bd2a0e29` |
| report.md | `397cf7a6bcdab860262767c28819a07023609acd3c01a3aede9f8a34cfd57c8a` |
| run-manifest.json | `03be2743761cc527fbf228548ec91ea4cb2fceb7e0b07d18f3f328322ac48a3f` |

Native in-memory save/reload was exact for all 8,824 rows; independent reload against exported CSV differs by at most 4.983625422061433e-10 (below 1e-7). Independent checks reconciled all forecast counts/MAE/MSE, every policy's net/gross equity, cost/delay stress, capital-time exposure, temporal P&L, all archive/artifact hashes and three actual price-reference trades.

Execution/probe evidence: `data/predictor_v0/final-fix-dev-003-evidence.json`. 

Holdout is unopened: no holdout ZIPs, labels, metrics or opening record. The offline replay's nominal Jan 1 exit is a forecast timestamp, not an observed holdout target. Historical boundary opens are idealized references, not executable fills; the trained-through-July-2025 model has no established current validity. Smallest next step: review this fixed development bundle. A separate forward/holdout experiment needs its own explicit scope.
