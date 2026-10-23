"""Reproducible fetch, development, and replay-forecast commands for predictor v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

# Set these before numerical imports, overriding oversized inherited defaults.
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
for _thread_variable in THREAD_ENVIRONMENT:
    os.environ[_thread_variable] = "4"

import numpy as np
import pandas as pd
import sklearn
import xgboost
from threadpoolctl import threadpool_info, threadpool_limits  # required by scikit-learn

from .data import (
    FEATURE_COLUMNS,
    SUPPORTED_SYMBOLS,
    _clean_and_validate,
    build_features,
    build_samples,
    download_archives,
    drop_incomplete_candles,
    load_candles,
    normalize_klines,
    validate_candles,
)
from .model import (
    SYMBOL_MAPPING,
    TARGET_ID,
    TREE_SETTINGS,
    ModelSchemaError,
    choose_action,
    evaluate_forecast_arms,
    fit_models,
    load_tree,
    net_return,
    save_tree,
    split_samples,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
ERROR_EXIT = 2
DECLARED_DATES = {
    "raw_start_utc": "2022-12-01T00:00:00Z",
    "sample_start_utc": "2023-01-01T00:00:00Z",
    "train_end_utc": "2025-07-01T00:00:00Z",
    "development_end_utc": "2026-01-01T00:00:00Z",
    "holdout_end_utc": "2026-09-01T00:00:00Z",
}


class CommandError(ValueError):
    """A user-facing failure with a stable machine-readable code."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@threadpool_limits.wrap(limits=4)
def main(
    argv: list[str] | None = None,
    *,
    transport: Callable[[str], bytes] | None = None,
    clock: Callable[[], pd.Timestamp] | None = None,
    repository_root: Path | None = None,
) -> int:
    """Run a command and return its process exit status; dependencies are injectable for tests."""
    parser = _parser()
    args = parser.parse_args(argv)
    root = Path(repository_root or REPOSITORY_ROOT).resolve()
    run_dir: Path | None = None
    now: pd.Timestamp | None = None
    fetch_diagnostics = {}
    started = time.perf_counter()
    try:
        run_dir = _run_dir(root, args.run_id) if args.command == "develop" else None
        config_path = _resolve_path(args.config, root)
        now = _utc_now(clock)
        config = _load_config(config_path)
        if args.command == "fetch":
            payload = _fetch(config, root, bool(args.include_holdout), transport, fetch_diagnostics)
        elif args.command == "develop":
            assert run_dir is not None
            payload = _develop(config, root, run_dir, now)
        else:
            payload = _forecast(
                config,
                _resolve_path(args.run_dir, root),
                args.symbol,
                _resolve_path(args.candles, root) if args.candles else None,
                args.as_of,
                now,
                transport,
            )
        _emit(payload)
        return 0
    except Exception as error:  # command boundaries must never expose a traceback on stdout
        code, detail = _error_details(error)
        if args.command == "develop" and run_dir is not None and code != "RUN_ALREADY_COMPLETED":
            failure_time = now if now is not None else pd.Timestamp(datetime.now(timezone.utc))
            _write_failed_manifest(run_dir, code, detail, failure_time, root)
        payload = {"status": "ERROR", "code": code, "action": "NO_TRADE"}
        if args.command == "fetch":
            diagnostics_path = root / "data" / "predictor_v0" / f"fetch-failure-{time.time_ns()}.json"
            _write_json(diagnostics_path, {
                **fetch_diagnostics, **payload, "error": _portable_detail(detail, root),
                "created_at_utc": now, "elapsed_seconds": time.perf_counter() - started,
            })
            payload["diagnostics_path"] = _relative_path(root, diagnostics_path)
        print(f"{code}: {detail}", file=sys.stderr)
        _emit(payload)
        return ERROR_EXIT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--config", required=True)
    fetch.add_argument("--include-holdout", action="store_true")
    develop = commands.add_parser("develop")
    develop.add_argument("--config", required=True)
    develop.add_argument("--run-id", required=True)
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--config", required=True)
    forecast.add_argument("--run-dir", required=True)
    forecast.add_argument("--symbol", required=True)
    forecast.add_argument("--candles")
    forecast.add_argument("--as-of")
    return parser


def _fetch(config: dict, root: Path, include_holdout: bool, transport, diagnostics: dict | None = None) -> dict:
    started = time.perf_counter()
    data_dir = root / "data" / "predictor_v0"
    raw_dir = data_dir / "raw"
    diagnostics = diagnostics if diagnostics is not None else {}
    start = _utc_config(config, "raw_start_utc")
    end_name = "holdout_end_utc" if include_holdout else "development_end_utc"
    end = _utc_config(config, end_name)
    diagnostics.update({"include_holdout": include_holdout, "requested_start_utc": start, "requested_end_utc": end})
    records = download_archives(config, raw_dir, include_holdout=include_holdout, transport=transport)
    manifest_records = [{**record, "path": _relative_path(root, Path(record["path"]))} for record in records]
    diagnostics["archives"] = manifest_records
    frames = []
    raw_close_time_anomalies = []
    for record in records:
        diagnostics["current_archive"] = record["url"]
        with zipfile.ZipFile(record["path"]) as archive:
            csv_names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
            if len(csv_names) != 1:
                raise CommandError("ARCHIVE_INVALID", f"expected exactly one CSV in {record['path']}")
            frame = pd.read_csv(archive.open(csv_names[0]), header=None)
        normalized = normalize_klines(frame, record["url"].split("/")[-3], record["unit"])
        raw_close_time_anomalies.extend({
            **anomaly, "archive_url": record["url"], "archive_period": record["period"],
            "archive_source_type": record["source_type"],
        } for anomaly in normalized.attrs.get("raw_close_time_anomalies", []))
        diagnostics["raw_close_time_anomalies"] = raw_close_time_anomalies
        period_start = pd.Timestamp(record["period"], tz="UTC")
        period_end = period_start + (pd.offsets.MonthBegin(1) if record["source_type"] == "monthly" else pd.Timedelta(days=1))
        validate_candles(normalized, period_start, period_end)
        frames.append(normalized)
    if not frames:
        raise CommandError("ARCHIVE_EMPTY", "no archive rows were downloaded")
    cleaned, raw_quality = _clean_and_validate(pd.concat(frames, ignore_index=True), None, None)
    requested = cleaned.loc[
        (cleaned["open_time_utc"] >= start) & (cleaned["open_time_utc"] < end)
    ].reset_index(drop=True)
    quality = validate_candles(requested, start, end)
    quality["duplicate_rows_dropped"] = raw_quality["duplicate_rows_dropped"]
    quality["raw_close_time_anomalies"] = raw_close_time_anomalies
    diagnostics["quality"] = quality
    missing_symbols = sorted(set(config["symbols"]) - set(quality["coverage"]))
    low_coverage = [symbol for symbol, report in quality["coverage"].items() if report["coverage"] < 0.99]
    if missing_symbols or low_coverage:
        names = ", ".join(missing_symbols + low_coverage)
        raise CommandError("RAW_COVERAGE_INSUFFICIENT", f"raw coverage below 99% for {names}")
    candles_path = data_dir / "candles.csv.gz"
    manifest = {
        "status": "COMPLETED",
        "include_holdout": include_holdout,
        "requested_start_utc": start,
        "requested_end_utc": end,
        "archives": manifest_records,
        "quality": quality,
    }
    # Stage the pair first; restore the previous pair if publication itself fails.
    with TemporaryDirectory(prefix=".fetch-", dir=data_dir) as temporary:
        staging = Path(temporary)
        names = ("candles.csv.gz", "data-manifest.json")
        _write_csv(requested, staging / names[0])
        manifest["hashes"] = {"config": _sha256_path(Path(config["_path"])), "candles": _sha256_path(staging / names[0])}
        manifest["elapsed_seconds"] = time.perf_counter() - started
        _write_json(staging / names[1], manifest)
        previous = {}
        for name in names:
            target = data_dir / name
            if target.exists():
                previous[name] = staging / f"previous-{name}"
                shutil.copyfile(target, previous[name])
        try:
            for name in names:
                (staging / name).replace(data_dir / name)
        except OSError:
            for name in names:
                if name in previous:
                    previous[name].replace(data_dir / name)
                else:
                    (data_dir / name).unlink(missing_ok=True)
            raise
    return {"status": "OK", "command": "fetch", "candles": _relative_path(root, candles_path), "quality": quality}


def _develop(config: dict, root: Path, run_dir: Path, now: pd.Timestamp) -> dict:
    if (run_dir / "run-manifest.json").is_file():
        existing = _read_json(run_dir / "run-manifest.json", "RUN_MANIFEST_INVALID")
        if existing.get("status") == "COMPLETED":
            raise CommandError("RUN_ALREADY_COMPLETED", f"run ID already completed: {run_dir.name}")
        raise CommandError("RUN_DIRECTORY_EXISTS", f"run directory already exists: {run_dir.name}")
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", _public_config(config))
    data_dir = root / "data" / "predictor_v0"
    candles_path = data_dir / "candles.csv.gz"
    manifest_path = data_dir / "data-manifest.json"
    if not candles_path.is_file() or not manifest_path.is_file():
        raise CommandError("FETCHED_DATA_REQUIRED", "run fetch before develop")
    fetched = _read_json(manifest_path, "DATA_MANIFEST_INVALID")
    if fetched.get("status") != "COMPLETED":
        raise CommandError("FETCHED_DATA_REQUIRED", "fetched data manifest is not complete")
    recorded_hashes = fetched.get("hashes")
    current_hashes = {
        "candles": _sha256_path(candles_path),
        "config": _sha256_path(Path(config["_path"])),
    }
    if not isinstance(recorded_hashes, dict) or any(recorded_hashes.get(name) != value for name, value in current_hashes.items()):
        raise CommandError("DATA_INTEGRITY_ERROR", "fetched candle or config hash does not match current files")
    candles = load_candles(candles_path)
    development_end = _utc_config(config, "development_end_utc")
    candles = candles.loc[candles["open_time_utc"] < development_end].reset_index(drop=True)
    samples = build_samples(candles, horizon_hours=int(config["horizon_hours"]))
    splits = split_samples(samples, config)
    train = splits["train"]
    development = splits["development"]
    if train.empty or development.empty:
        raise CommandError("INSUFFICIENT_DEVELOPMENT_DATA", "train and development splits must both contain labeled rows")
    schema = {"feature_columns": FEATURE_COLUMNS, "symbol_mapping": SYMBOL_MAPPING, "target_id": TARGET_ID}
    schema_path = run_dir / "feature-schema.json"
    _write_json(schema_path, schema)
    hashes = {
        "data": _sha256_path(candles_path),
        "config": _sha256_path(Path(config["_path"])),
        "code": _code_hash(),
        "feature_schema": _sha256_path(schema_path),
    }
    models = fit_models(train, FEATURE_COLUMNS, config)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "symbol_mapping": SYMBOL_MAPPING,
        "target_id": TARGET_ID,
        "training_cutoff_utc": _iso(_utc_config(config, "train_end_utc")),
        "seed": 42,
        "estimator_settings": TREE_SETTINGS,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "hashes": hashes,
    }
    model_path = run_dir / "model.json"
    save_tree(models["tree"], model_path, metadata)
    generated_metadata = model_path.with_suffix(".metadata.json")
    metadata_path = run_dir / "model-metadata.json"
    generated_metadata.replace(metadata_path)
    restored = load_tree(model_path, metadata_path)
    original_prediction = models["tree"].predict(development[FEATURE_COLUMNS])
    restored_prediction = restored.predict(development[FEATURE_COLUMNS])
    reload_passed = bool(np.allclose(original_prediction, restored_prediction, rtol=0.0, atol=0.0))
    if not reload_passed:
        raise CommandError("MODEL_RELOAD_MISMATCH", "reloaded native tree predictions differ")
    development_forecasts = {
        "tree": restored_prediction,
        "ridge": models["ridge"].predict(development[FEATURE_COLUMNS]),
        "zero": models["zero"].predict(development[FEATURE_COLUMNS]),
    }
    comparison = evaluate_forecast_arms(development, development_forecasts, config)
    predictions = development.copy()
    for name, values in development_forecasts.items():
        predictions[f"{name}_forecast_gross_return"] = values
    _write_csv(predictions, run_dir / "development-predictions.csv.gz")
    combined = pd.concat(
        [ledger.assign(policy_name=name) for name, ledger in comparison["ledgers"].items()],
        ignore_index=True,
    )
    _write_csv(combined, run_dir / "development-opportunities.csv.gz")
    metrics = {
        "development": {
            "forecast_metrics": comparison["forecast_metrics"],
            "policy_metrics": comparison["policy_metrics"],
            "strategy_summaries": comparison["strategy_summaries"],
            "screening": _screen_development(comparison, all(config[key] == value for key, value in DECLARED_DATES.items())),
            "population": {
                "valid_purged_forecast_rows": len(development),
                "rows_without_valid_delay_stress": int((~comparison["ledgers"]["tree"]["delayed_valid"]).sum()),
                "delay_excluded_opportunities": comparison["strategy_summaries"]["tree"]["delayed_reference_price_proxy"]["excluded_opportunity_count"],
            },
            "units": {"returns": "fractions; 0.01 = 1%; 1 bp = 0.0001", "equity": "USD", "capital_time": "USD-hours"},
        }
    }
    _write_json(run_dir / "metrics.json", metrics)
    train_start = _utc_config(config, "sample_start_utc")
    train_end = _utc_config(config, "train_end_utc")
    development_end = _utc_config(config, "development_end_utc")
    thread_limits = {"fitted_tree": _thread_limits(models["tree"]), "reloaded_tree": _thread_limits(restored)}
    elapsed_wall = time.perf_counter() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    screening = metrics["development"]["screening"]
    strategies = comparison["strategy_summaries"]
    passive = strategies["buy_and_hold"]
    report_lines = [
        "# Predictor v0 development report",
        "",
        "Status: DEVELOPMENT_ONLY (development-only). Labels and metrics below are for the development population only.",
        f"Overall screening label: {screening['label']}. This is a heuristic screen, not a significance claim.",
        f"Run: {run_dir.name}; created: {_iso(now)}; wall evaluation time: {elapsed_wall:.6f}s; process CPU: {elapsed_cpu:.6f}s.",
        f"Train boundary: [{_iso(train_start)}, {_iso(train_end)}) with {len(train)} labeled rows.",
        f"Development boundary: [{_iso(train_end)}, {_iso(development_end)}) with {len(development)} labeled rows.",
        "",
        "All features use only completed hourly candles through each feature cutoff, with 50-bar warmup and no window crossing a gap. Targets use the next open and the open four hours later; split labels finish strictly before their block end.",
        "Declared forecast arms: tree, Ridge, and zero; cash, trend and buy-and-hold have no quantitative forecast metrics.",
        "Metrics summary (pooled forecast count/MAE/MSE):",
        *[
            f"{name}: {comparison['forecast_metrics'][name]['pooled']['count']} / "
            f"{comparison['forecast_metrics'][name]['pooled']['mae']} / "
            f"{comparison['forecast_metrics'][name]['pooled']['mse']}"
            for name in ("tree", "ridge", "zero")
        ],
        "",
        "| Policy | Opportunities | Selected | Mean net | Gross cumulative | Net cumulative | Sampled drawdown | Capital-time exposure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {name} | {s['all_opportunities']} | {s['selected_opportunities']} | {s['average_net_return']} | {s['gross_cumulative_proxy_return']} | {s['cumulative_proxy_return']} | {s['sampled_drawdown']} | {s['exposure']['capital_time_fraction']} |" for name, s in strategies.items()],
        "",
        "Returns are fractions (0.01 = 1%), equity is USD, capital-time is USD-hours. Two independent $5,000 sleeves reinvest after expiry. Net cumulative return is their summed ending equity / $10,000 - 1. Gross uses the same selections without costs.",
        "Capital-time exposure divides invested sleeve USD-hours by all available sleeve USD-hours over the reported span; capital is marked at realized exits. Cash time and gaps remain in the denominator. Selected-row fraction is a separate diagnostic.",
        "",
        f"Buy-and-hold reference span: {passive['span_start_utc']} through {passive['span_end_utc']}; {passive['span_selection']}.",
        passive["exposure_caveat"],
        "Policy returns on this same span: " + json.dumps(passive["same_span_policy_returns"], sort_keys=True),
        "",
        "Base and doubled costs preserve policy selections. Delay shifts both entry and exit one hour and preserves four-hour holds; base/delay comparisons alone use their common valid subset. Both cumulative stresses use independent sleeves.",
        "",
        "Screening criteria:",
        *[f"- {name}: {'PASS' if criterion['passed'] else 'FAIL'}; observed={criterion['observed']}; required={criterion['required']}" for name, criterion in screening["criteria"].items()],
        "",
        "| Tree entry month | Selected trades | Share of tree trades | Net sleeve P&L (USD) |",
        "|---|---:|---:|---:|",
        *[f"| {month} | {value['selected_trades']} | {value['fraction_of_selected_trades']} | {value['net_pnl_usd']} |" for month, value in strategies["tree"]["temporal_concentration"]["months"].items()],
        "",
        "Historical results are reference_price_proxy estimates. Boundary opens precede actual inference and do not prove executable fills. Sampled drawdown misses intrablock losses. Fresh inspection is PAPER_ONLY and research_only; a live source error is SOURCE_UNAVAILABLE, never a synthetic forecast.",
        "",
        "Provenance hashes: " + json.dumps({**hashes, "model": _sha256_path(model_path), "data_manifest": _sha256_path(manifest_path)}, sort_keys=True),
        "Measured thread limits: " + json.dumps(thread_limits, sort_keys=True),
        "Source quality (including every accepted raw-close anomaly): " + json.dumps(fetched["quality"], sort_keys=True),
        "Archive units: " + json.dumps({unit: sum(record["unit"] == unit for record in fetched["archives"]) for unit in ("ms", "us")}),
        "The shared data-manifest.json and candles.csv.gz live under data/predictor_v0; run artifacts live under artifacts/predictor_v0/" + run_dir.name + "/.",
        "",
        "Holdout was not opened for this ordinary development run; no holdout metrics and no holdout labels are included.",
        "Smallest next step: review this fixed development bundle; any separate forward or holdout evaluation requires its own authorization.",
        "",
        "Complete metrics, including per-asset results, stress comparisons and temporal concentration for every policy:",
        "```json", json.dumps(metrics, indent=2, sort_keys=True), "```",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (run_dir / "agent-log.md").write_text(
        f"# Predictor v0 agent log\n\nCompleted development run `{run_dir.name}` at {_iso(now)}.\n"
        f"Command: python -m experiments.predictor_v0.run develop --config experiments/predictor_v0/config.json --run-id {run_dir.name}\n"
        f"Screen: {screening['label']}. Holdout unopened; code and config hashes are in run-manifest.json.\n",
        encoding="utf-8",
    )
    run_manifest = {
        "status": "COMPLETED",
        "run_id": run_dir.name,
        "created_at_utc": now,
        "training_end_utc": _utc_config(config, "train_end_utc"),
        "development_end_utc": _utc_config(config, "development_end_utc"),
        "hashes": {**hashes, "model": _sha256_path(model_path), "model_metadata": _sha256_path(metadata_path)},
        "data_manifest_hash": _sha256_path(manifest_path),
        "cpu": {"max_threads": config["max_threads"], "processor": platform.processor() or "unknown", "cpu_count": os.cpu_count(), "actual_limits": thread_limits},
        "elapsed_seconds": elapsed_wall,
        "process_seconds": elapsed_cpu,
        "timing_scope": "data loading through evaluation and metrics persistence, excluding final report/manifest writes",
        "reload_verification": {"passed": reload_passed, "rows": int(len(development))},
        "sample_rows": {"train": int(len(train)), "development": int(len(development))},
        "feature_excluded_rows": int(len(candles) - len(samples)),
        "screening_label": screening["label"],
        "versions": metadata["versions"],
        "artifact_sha256": {
            path.name: _sha256_path(path) for path in sorted(run_dir.iterdir()) if path.is_file()
        },
        "paths": {
            "run_dir": _relative_path(root, run_dir),
            "config": _relative_path(root, run_dir / "config.json"),
            "data_candles": _relative_path(root, candles_path),
            "data_manifest": _relative_path(root, manifest_path),
            "model": _relative_path(root, model_path),
            "model_metadata": _relative_path(root, metadata_path),
            "feature_schema": _relative_path(root, schema_path),
            "development_predictions": _relative_path(root, run_dir / "development-predictions.csv.gz"),
            "development_opportunities": _relative_path(root, run_dir / "development-opportunities.csv.gz"),
        },
    }
    _write_json(run_dir / "run-manifest.json", run_manifest)
    return {"status": "OK", "command": "develop", "run_dir": _relative_path(root, run_dir), "action": "NO_TRADE", "screening_label": screening["label"], "training_end_utc": _iso(train_end), "development_end_utc": _iso(development_end)}


def _screen_development(comparison: dict, full_declared_run: bool) -> dict:
    tree = comparison["strategy_summaries"]["tree"]
    forecast = comparison["forecast_metrics"]
    counts = {symbol: tree["per_asset_metrics"][symbol]["selected_trades"] for symbol in SYMBOL_MAPPING}
    base = {symbol: tree["per_asset_metrics"][symbol]["average_net_return"] for symbol in SYMBOL_MAPPING}
    doubled = {symbol: tree["doubled_costs_fixed_selections"]["per_asset_metrics"][symbol]["average_net_return"] for symbol in SYMBOL_MAPPING}
    mse = {arm: forecast[arm]["pooled"]["mse"] for arm in ("tree", "ridge", "zero")}
    delay = tree["delayed_reference_price_proxy"]["average_net_return"]
    positive = lambda value: value is not None and value > 0.0
    criteria = {
        "full_declared_development": {"passed": full_declared_run, "observed": full_declared_run, "required": True},
        "pooled_selected_count": {"passed": tree["selected_opportunities"] >= 100, "observed": tree["selected_opportunities"], "required": ">= 100"},
        "per_asset_selected_count": {"passed": all(count >= 25 for count in counts.values()), "observed": counts, "required": ">= 25 each"},
        "tree_mse_below_zero_and_ridge": {"passed": all(value is not None for value in mse.values()) and mse["tree"] < min(mse["zero"], mse["ridge"]), "observed": mse, "required": "tree < both"},
        "base_mean_net_positive_per_asset": {"passed": all(map(positive, base.values())), "observed": base, "required": "> 0 each"},
        "doubled_mean_net_positive_per_asset": {"passed": all(map(positive, doubled.values())), "observed": doubled, "required": "> 0 each"},
        "delayed_mean_net_positive_pooled": {"passed": positive(delay), "observed": delay, "required": "> 0"},
    }
    if not full_declared_run:
        label = "PIPELINE_ONLY"
    elif not criteria["pooled_selected_count"]["passed"] or not criteria["per_asset_selected_count"]["passed"]:
        label = "INSUFFICIENT_EVIDENCE"
    else:
        label = "CANDIDATE_FOR_FORWARD_TEST" if all(item["passed"] for item in criteria.values()) else "NO_EDGE"
    return {"label": label, "criteria": criteria, "failed_criteria": [name for name, item in criteria.items() if not item["passed"]], "interpretation": "Fixed heuristic screening thresholds; not statistical significance or proof of executable profitability."}


def _forecast(config: dict, run_dir: Path, symbol: str, candle_arg: str | None, as_of_arg: str | None, now: pd.Timestamp, transport) -> dict:
    if as_of_arg is not None and candle_arg is None:
        raise CommandError("REPLAY_REQUIRES_CANDLES", "--as-of requires --candles for an explicit offline replay")
    if symbol not in SUPPORTED_SYMBOLS:
        raise CommandError("UNSUPPORTED_SYMBOL", f"unsupported symbol: {symbol}")
    if symbol not in config.get("symbols", []):
        raise CommandError("UNSUPPORTED_SYMBOL", f"symbol is not configured: {symbol}")
    as_of = _utc_timestamp(as_of_arg) if as_of_arg is not None else now
    if candle_arg is not None:
        try:
            candles = load_candles(Path(candle_arg), as_of_utc=as_of)
        except (OSError, ValueError) as error:
            raise CommandError("INVALID_CANDLES", str(error)) from error
        replay = True
    else:
        candles = _download_live_candles(symbol, transport)
        candles = drop_incomplete_candles(candles, as_of)
        candles = candles.loc[candles["symbol"].eq(symbol)].tail(250).reset_index(drop=True)
        replay = False
    latest_expected = as_of.floor("h")
    symbol_candles = candles.loc[candles["symbol"].eq(symbol)]
    if symbol_candles.empty:
        raise CommandError("INVALID_CANDLES", f"no valid candles for {symbol}")
    latest_actual = symbol_candles["close_time_utc"].max()
    if latest_actual != latest_expected:
        grace = latest_expected + pd.Timedelta(minutes=2)
        code = "DATA_PENDING" if as_of <= grace else "STALE_DATA"
        raise CommandError(code, f"latest completed cutoff is {_iso(latest_actual)}, expected {_iso(latest_expected)}")
    try:
        features = build_features(candles, as_of_utc=as_of)
    except (OSError, ValueError) as error:
        raise CommandError("INVALID_CANDLES", str(error)) from error
    try:
        latest_features = features.loc[[(symbol, latest_expected)], FEATURE_COLUMNS]
    except KeyError as error:
        raise CommandError("INSUFFICIENT_WARMUP", "latest completed candle lacks causal feature warmup") from error
    if len(latest_features) != 1:
        raise CommandError("INVALID_CANDLES", "latest feature row is not unique")
    model_path = run_dir / "model.json"
    metadata_path = run_dir / "model-metadata.json"
    schema_path = run_dir / "feature-schema.json"
    run_manifest = _read_json(run_dir / "run-manifest.json", "MODEL_INTEGRITY_ERROR")
    if run_manifest.get("status") != "COMPLETED":
        raise CommandError("MODEL_INTEGRITY_ERROR", "run manifest is not completed")
    recorded_hashes = run_manifest.get("hashes")
    if not model_path.is_file() or not metadata_path.is_file() or not schema_path.is_file() or not isinstance(recorded_hashes, dict):
        raise CommandError("MODEL_INTEGRITY_ERROR", "run manifest is missing model integrity records")
    current_hashes = {
        "config": _sha256_path(Path(config["_path"])),
        "code": _code_hash(),
        "model": _sha256_path(model_path),
        "model_metadata": _sha256_path(metadata_path),
        "feature_schema": _sha256_path(schema_path),
    }
    if any(recorded_hashes.get(name) != current_hashes[name] for name in ("model", "feature_schema", "config", "code")):
        raise CommandError("MODEL_INTEGRITY_ERROR", "current config, code, model or feature schema does not match the completed run")
    schema = _read_json(schema_path, "MODEL_SCHEMA_ERROR")
    if schema != {"feature_columns": FEATURE_COLUMNS, "symbol_mapping": SYMBOL_MAPPING, "target_id": TARGET_ID}:
        raise CommandError("MODEL_SCHEMA_ERROR", "feature schema does not match predictor v0")
    model = load_tree(model_path, metadata_path)
    metadata = _read_json(metadata_path, "MODEL_SCHEMA_ERROR")
    if recorded_hashes.get("model_metadata") != current_hashes["model_metadata"]:
        raise CommandError("MODEL_INTEGRITY_ERROR", "model metadata hash does not match run manifest")
    if any(recorded_hashes.get(key) != value for key, value in metadata["hashes"].items()):
        raise CommandError("MODEL_INTEGRITY_ERROR", "model metadata provenance does not match the completed run")
    if _utc_timestamp(metadata["training_cutoff_utc"]) != _utc_config(config, "train_end_utc"):
        raise CommandError("MODEL_INTEGRITY_ERROR", "training cutoff does not match the completed run")
    gross = float(model.predict(latest_features)[0])
    net = float(net_return(gross, config["fee_bps_per_side"], config["slippage_bps_per_side"]))
    if latest_expected.hour not in config["opportunity_hours_utc"]:
        action, reason = "NO_TRADE", "OUTSIDE_OPPORTUNITY_WINDOW"
    elif as_of - latest_expected > pd.Timedelta(minutes=5):
        action, reason = "NO_TRADE", "SIGNAL_EXPIRED"
    else:
        action = choose_action(gross, config)
        reason = "MODEL_EDGE" if action == "LONG" else "EDGE_BELOW_MARGIN"
    training_end = _utc_timestamp(metadata["training_cutoff_utc"])
    return {
        "status": "PAPER_ONLY",
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "symbol": symbol,
        "feature_cutoff_utc": latest_expected,
        "created_at_utc": now,
        "reference_entry_utc": latest_expected,
        "reference_exit_utc": latest_expected + pd.Timedelta(hours=int(config["horizon_hours"])),
        "training_end_utc": training_end,
        "model_age_days": float((as_of - training_end).total_seconds() / 86_400.0),
        "research_only": True,
        "target_id": TARGET_ID,
        "forecast_gross_return": gross,
        "forecast_net_return_assumption": net,
        "action": action,
        "reason": reason,
        "model_sha256": current_hashes["model"],
        "feature_schema_sha256": current_hashes["feature_schema"],
        "replay": replay,
        "thread_limits": _thread_limits(model),
    }


def _download_live_candles(symbol: str, transport) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines?" + urlencode({"symbol": symbol, "interval": "1h", "limit": 251})
    try:
        if transport is not None:
            payload = transport(url)
        else:
            with urlopen(url, timeout=30) as response:  # nosec B310 - fixed public Binance endpoint
                payload = response.read()
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, list):
            raise CommandError("SOURCE_UNAVAILABLE", "public Binance response was not a kline list")
        frame = pd.DataFrame(decoded)
    except CommandError:
        raise
    except Exception as error:
        raise CommandError("SOURCE_UNAVAILABLE", "unable to retrieve public Binance candles") from error
    try:
        return normalize_klines(frame, symbol, "ms")
    except ValueError as error:
        raise CommandError("INVALID_CANDLES", str(error)) from error


def _load_config(path: Path) -> dict:
    config = _read_json(path, "CONFIG_INVALID")
    if not isinstance(config, dict):
        raise CommandError("CONFIG_INVALID", "config must be a JSON object")
    fixed = {
        "experiment_id": "crypto_4h_tree_v0", "symbols": ["BTCUSDT", "ETHUSDT"],
        "venue": "binance_spot", "interval": "1h", "horizon_hours": 4,
        "fee_bps_per_side": 10.0, "slippage_bps_per_side": 5.0,
        "minimum_predicted_net_return": 0.001, "opportunity_hours_utc": [0, 4, 8, 12, 16, 20],
        "seed": 42, "max_threads": 4, "initial_equity": 10000.0, "mode": "paper_only",
    }
    if set(config) != set(fixed) | set(DECLARED_DATES):
        raise CommandError("CONFIG_INVALID", "config keys must match the fixed predictor contract")
    for key, expected in fixed.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise CommandError("CONFIG_INVALID", f"fixed predictor contract mismatch: {key}")
    dates = [_utc_config(config, key) for key in DECLARED_DATES]
    if any(pd.isna(value) or value != value.floor("h") for value in dates) or dates != sorted(set(dates)):
        raise CommandError("CONFIG_INVALID", "dates must be distinct, increasing UTC hour boundaries")
    config["_path"] = str(path.resolve())
    return config


def _thread_limits(tree) -> dict:
    native = json.loads(tree.get_booster().save_config())["learner"]["generic_param"]
    return {
        "environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "tree_nthread": int(native["nthread"]),
        "numerical_libraries": [
            {key: pool[key] for key in ("internal_api", "prefix", "num_threads")}
            for pool in threadpool_info()
        ],
    }


def _run_dir(root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise CommandError("RUN_ID_INVALID", "run ID must be a simple directory name")
    return root / "artifacts" / "predictor_v0" / run_id


def _resolve_path(value: str, root: Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise CommandError("PATH_INVALID", f"internal artifact is outside repository root: {path}") from error


def _read_json(path: Path, code: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CommandError(code, f"unreadable JSON: {path}") from error
    if not isinstance(value, dict):
        raise CommandError(code, f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def _write_failed_manifest(run_dir: Path, code: str, detail: str, now: pd.Timestamp, root: Path) -> None:
    manifest_path = run_dir / "run-manifest.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("status") == "COMPLETED":
            return
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run-manifest.json",
        {"status": "ERROR", "code": code, "error": _portable_detail(detail, root), "created_at_utc": now},
    )


def _portable_detail(detail: str, root: Path) -> str:
    text = str(detail)
    root_text = str(root.resolve())
    return text.replace(root_text + "\\", "").replace(root_text + "/", "").replace(root_text, ".")


def _error_details(error: Exception) -> tuple[str, str]:
    if isinstance(error, CommandError):
        return error.code, error.detail
    if isinstance(error, ModelSchemaError):
        return error.code, str(error)
    detail = str(error) or type(error).__name__
    if "unsupported symbol" in detail.lower():
        return "UNSUPPORTED_SYMBOL", detail
    if "candle" in detail.lower() or "ohlcv" in detail.lower():
        return "INVALID_CANDLES", detail
    return "COMMAND_FAILED", detail


def _public_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "_path"}


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now(clock: Callable[[], pd.Timestamp] | None) -> pd.Timestamp:
    value = clock() if clock is not None else pd.Timestamp(datetime.now(timezone.utc))
    return _utc_timestamp(value)


def _utc_config(config: dict, name: str) -> pd.Timestamp:
    return _utc_timestamp(config[name])


def _utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise CommandError("TIME_INVALID", "timestamps must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def _iso(value: pd.Timestamp) -> str:
    return _utc_timestamp(value).isoformat().replace("+00:00", "Z")


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return _iso(pd.Timestamp(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _emit(payload: dict) -> None:
    print(json.dumps(payload, default=_json_default, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
