"""Fixed chronological predictor arms and reference-price policy evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from .data import FEATURE_COLUMNS
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits


TARGET_ID = "next_hour_open_to_open_plus_4h"
SYMBOL_MAPPING = {"BTCUSDT": 0, "ETHUSDT": 1}
OPPORTUNITY_HOURS = (0, 4, 8, 12, 16, 20)
INITIAL_EQUITY = 10_000.0
MINIMUM_PREDICTED_NET_RETURN = 0.001
TREE_SETTINGS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "random_state": 42,
    "n_jobs": 4,
}
_METADATA_KEYS = {
    "feature_columns",
    "symbol_mapping",
    "target_id",
    "training_cutoff_utc",
    "seed",
    "estimator_settings",
    "versions",
    "hashes",
}
_VERSION_KEYS = {"numpy", "pandas", "scikit_learn", "xgboost"}
_HASH_KEYS = {"data", "config", "code", "feature_schema"}


class ModelSchemaError(ValueError):
    """Raised when a persisted tree cannot prove its input schema."""

    def __init__(self, detail: str):
        super().__init__(f"MODEL_SCHEMA_ERROR: {detail}")
        self.code = "MODEL_SCHEMA_ERROR"


class _ZeroModel:
    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


def split_samples(samples: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Chronologically split samples using strict, label-purged UTC boundaries."""
    required = {"symbol", "feature_cutoff_utc", "label_end_utc", "target_valid", "target_return"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"samples missing required columns: {', '.join(missing)}")
    frame = samples.copy()
    frame["feature_cutoff_utc"] = _utc_series(frame["feature_cutoff_utc"], "feature_cutoff_utc")
    frame["label_end_utc"] = _utc_series(frame["label_end_utc"], "label_end_utc")
    boundaries = {
        name: _utc_config(config, name)
        for name in ("sample_start_utc", "train_end_utc", "development_end_utc", "holdout_end_utc")
    }
    if not (
        boundaries["sample_start_utc"] < boundaries["train_end_utc"]
        < boundaries["development_end_utc"] < boundaries["holdout_end_utc"]
    ):
        raise ValueError("sample split boundaries must be strictly increasing")
    valid_label = frame["target_valid"].eq(True) & frame["target_return"].notna()
    masks = {
        "train": (
            (frame["feature_cutoff_utc"] >= boundaries["sample_start_utc"])
            & (frame["feature_cutoff_utc"] < boundaries["train_end_utc"])
            & (frame["label_end_utc"] < boundaries["train_end_utc"])
        ),
        "development": (
            (frame["feature_cutoff_utc"] >= boundaries["train_end_utc"])
            & (frame["feature_cutoff_utc"] < boundaries["development_end_utc"])
            & (frame["label_end_utc"] < boundaries["development_end_utc"])
        ),
        "holdout": (
            (frame["feature_cutoff_utc"] >= boundaries["development_end_utc"])
            & (frame["feature_cutoff_utc"] < boundaries["holdout_end_utc"])
            & (frame["label_end_utc"] < boundaries["holdout_end_utc"])
        ),
    }
    return {
        name: frame.loc[mask & valid_label].sort_values(["feature_cutoff_utc", "symbol"], kind="stable").reset_index(drop=True)
        for name, mask in masks.items()
    }


def fit_models(train: pd.DataFrame, feature_columns: list[str], config: dict) -> dict:
    """Fit only the fixed Ridge, fixed tree, and zero baseline forecast arms."""
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError("feature columns must exactly match FEATURE_COLUMNS")
    missing = sorted(set(feature_columns + ["target_return"]) - set(train.columns))
    if missing:
        raise ValueError(f"training data missing required columns: {', '.join(missing)}")
    features = train.loc[:, feature_columns].to_numpy(dtype=float)
    target = train["target_return"].to_numpy(dtype=float)
    if len(train) == 0 or not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("training features and target must be finite and non-empty")
    ridge = Pipeline([("standardscaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    tree = XGBRegressor(**TREE_SETTINGS)
    with threadpool_limits(limits=4):
        ridge.fit(train.loc[:, feature_columns], target)
        tree.fit(train.loc[:, feature_columns], target)
    return {"zero": _ZeroModel(), "ridge": ridge, "tree": tree}


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return JSON-safe regression metrics, including explicit undefined reasons."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.ndim != 1 or predicted_values.ndim != 1 or len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must be equal-length one-dimensional arrays")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("actual and predicted must be finite")
    if len(actual_values) == 0:
        return {
            "count": 0, "mae": None, "mse": None, "mean_prediction": None, "mean_target": None,
            "prediction_std": None, "pearson_correlation": None, "pearson_reason": "INSUFFICIENT_SAMPLES",
            "oos_r2_vs_zero": None, "oos_r2_reason": "INSUFFICIENT_SAMPLES", "direction_agreement": None,
        }
    error = predicted_values - actual_values
    target_energy = float(np.dot(actual_values, actual_values))
    pearson_reason = None
    pearson = None
    if len(actual_values) < 2:
        pearson_reason = "INSUFFICIENT_SAMPLES"
    elif np.std(actual_values) == 0.0 or np.std(predicted_values) == 0.0:
        pearson_reason = "ZERO_VARIANCE"
    else:
        pearson = float(np.corrcoef(actual_values, predicted_values)[0, 1])
    return {
        "count": int(len(actual_values)),
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "mean_prediction": float(np.mean(predicted_values)),
        "mean_target": float(np.mean(actual_values)),
        "prediction_std": float(np.std(predicted_values)),
        "pearson_correlation": pearson,
        "pearson_reason": pearson_reason,
        "oos_r2_vs_zero": None if target_energy == 0.0 else float(1.0 - np.dot(error, error) / target_energy),
        "oos_r2_reason": "ZERO_TARGET_ENERGY" if target_energy == 0.0 else None,
        "direction_agreement": float(np.mean(np.sign(actual_values) == np.sign(predicted_values))),
    }


def save_tree(model, path: Path, metadata: dict) -> None:
    """Save only native XGBoost JSON plus an independently validated metadata file."""
    _validate_metadata(metadata)
    _validate_tree_settings(model, metadata["estimator_settings"])
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("tree path must use the native .json format")
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(destination))
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_tree(path: Path, metadata_path: Path):
    """Load a native tree only when its stored metadata matches the fixed contract."""
    model_path = Path(path)
    schema_path = Path(metadata_path)
    if not model_path.is_file():
        raise ModelSchemaError("missing native tree JSON")
    if not schema_path.is_file():
        raise ModelSchemaError("missing metadata JSON")
    try:
        stored = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelSchemaError("unreadable metadata JSON") from error
    _validate_metadata(stored)
    tree = XGBRegressor(**TREE_SETTINGS)
    try:
        tree.load_model(str(model_path))
    except xgboost.core.XGBoostError as error:
        raise ModelSchemaError("unreadable native tree JSON") from error
    return tree


def net_return(gross_return: float, fee_bps: float, slippage_bps: float) -> float:
    """Apply the declared round-trip reference-price fee and slippage policy."""
    try:
        gross_return = float(gross_return)
        fee_bps = float(fee_bps)
        slippage_bps = float(slippage_bps)
    except (TypeError, ValueError) as error:
        raise ValueError("gross return and costs must be numeric") from error
    if not all(math.isfinite(value) for value in (gross_return, fee_bps, slippage_bps)):
        raise ValueError("gross return and costs must be finite")
    if gross_return <= -1.0:
        raise ValueError("gross return must be greater than -1")
    if not (0.0 <= fee_bps / 10_000.0 < 1.0) or not (0.0 <= slippage_bps / 10_000.0 < 1.0):
        raise ValueError("costs must satisfy 0 <= bps / 10000 < 1")
    fee = fee_bps / 10_000.0
    slippage = slippage_bps / 10_000.0
    return (1.0 + gross_return) * (1.0 - slippage) * (1.0 - fee) / ((1.0 + slippage) * (1.0 + fee)) - 1.0


def choose_action(predicted_gross_return: float, config: dict) -> str:
    """Choose LONG only above the strictly positive cost-adjusted policy threshold."""
    _validate_policy_config(config)
    fee_bps, slippage_bps = _costs(config)
    return "LONG" if net_return(predicted_gross_return, fee_bps, slippage_bps) > MINIMUM_PREDICTED_NET_RETURN else "NO_TRADE"


def evaluate_opportunities(samples: pd.DataFrame, forecasts: np.ndarray, config: dict) -> tuple[pd.DataFrame, dict]:
    """Evaluate one forecast arm with fixed common opportunities and independent sleeves."""
    _validate_policy_config(config)
    required = {
        "symbol", "feature_cutoff_utc", "reference_entry_utc", "reference_exit_utc",
        "reference_entry_price", "reference_exit_price", "delayed_entry_utc", "delayed_exit_utc",
        "delayed_entry_price", "delayed_exit_price", "target_valid", "price_vs_sma20", "return_24h",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"samples missing required columns: {', '.join(missing)}")
    forecast_values = np.asarray(forecasts, dtype=float)
    if forecast_values.ndim != 1 or len(forecast_values) != len(samples) or not np.isfinite(forecast_values).all():
        raise ValueError("forecasts must be finite and align one-to-one with samples")
    frame = samples.copy().reset_index(drop=True)
    for column in ("feature_cutoff_utc", "reference_entry_utc", "reference_exit_utc", "delayed_entry_utc", "delayed_exit_utc"):
        frame[column] = _utc_series(frame[column], column)
    for column in ("reference_entry_price", "reference_exit_price", "delayed_entry_price", "delayed_exit_price"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    reference_timestamps = frame[["reference_entry_utc", "reference_exit_utc"]].notna().all(axis=1)
    if (reference_timestamps & ((frame["reference_exit_utc"] - frame["reference_entry_utc"]) != pd.Timedelta(hours=4))).any():
        raise ValueError("reference offsets must be exactly four hours")
    reference_offsets_match = reference_timestamps & (
        (frame["reference_exit_utc"] - frame["reference_entry_utc"]) == pd.Timedelta(hours=4)
    )
    valid_reference = (
        frame["target_valid"].eq(True)
        & reference_offsets_match
        & np.isfinite(frame[["reference_entry_price", "reference_exit_price"]]).all(axis=1)
        & (frame[["reference_entry_price", "reference_exit_price"]] > 0.0).all(axis=1)
    )
    delayed_timestamps = frame[["delayed_entry_utc", "delayed_exit_utc"]].notna().all(axis=1)
    # An immature delay reference is an exclusion from stress, not from forecasting.
    for delayed, reference in (("delayed_entry_utc", "reference_entry_utc"), ("delayed_exit_utc", "reference_exit_utc")):
        present = frame[delayed].notna() & frame[reference].notna()
        if (present & frame[delayed].ne(frame[reference] + pd.Timedelta(hours=1))).any():
            raise ValueError("reference offsets must use one-hour delayed entry and exit")
    delayed_valid = (
        valid_reference
        & delayed_timestamps
        & np.isfinite(frame[["delayed_entry_price", "delayed_exit_price"]]).all(axis=1)
        & (frame[["delayed_entry_price", "delayed_exit_price"]] > 0.0).all(axis=1)
    )
    if "development_end_utc" in config:
        delayed_valid &= frame["delayed_exit_utc"] < _utc_config(config, "development_end_utc")
    fee_bps, slippage_bps = _costs(config)
    frame["forecast_gross_return"] = forecast_values
    frame["valid_reference"] = valid_reference
    frame["delayed_valid"] = delayed_valid
    frame["is_opportunity"] = frame["valid_reference"] & frame["reference_entry_utc"].dt.hour.isin(OPPORTUNITY_HOURS)
    frame["common_valid_for_delay"] = frame["is_opportunity"] & frame["delayed_valid"]
    frame["action"] = "NO_TRADE"
    frame.loc[frame["is_opportunity"], "action"] = [
        choose_action(value, config) for value in frame.loc[frame["is_opportunity"], "forecast_gross_return"]
    ]
    frame["trend_action"] = "NO_TRADE"
    trend = frame["is_opportunity"] & (frame["price_vs_sma20"] > 0.0) & (frame["return_24h"] > 0.0)
    frame.loc[trend, "trend_action"] = "LONG"
    frame["cash_action"] = "NO_TRADE"
    frame = frame.sort_values(["reference_entry_utc", "symbol"], kind="stable").reset_index(drop=True)
    frame["selected"] = _nonoverlapping_selection(frame, frame["action"].eq("LONG"))
    frame["trend_selected"] = _nonoverlapping_selection(frame, frame["trend_action"].eq("LONG"))
    frame["reference_price_proxy_gross_return"] = np.where(
        frame["valid_reference"], frame["reference_exit_price"] / frame["reference_entry_price"] - 1.0, np.nan
    )
    frame["reference_price_proxy_net_return"] = _net_series(
        frame["reference_price_proxy_gross_return"], frame["selected"], fee_bps, slippage_bps
    )
    frame["delayed_reference_price_proxy_gross_return"] = np.where(
        frame["delayed_valid"], frame["delayed_exit_price"] / frame["delayed_entry_price"] - 1.0, np.nan
    )
    frame["delayed_reference_price_proxy_net_return"] = _net_series(
        frame["delayed_reference_price_proxy_gross_return"], frame["selected"] & frame["delayed_valid"], fee_bps, slippage_bps
    )
    frame, primary = _apply_sleeves(frame, "selected", "reference_price_proxy_gross_return", fee_bps, slippage_bps)
    trend_ledger, trend_summary = _apply_sleeves(frame, "trend_selected", "reference_price_proxy_gross_return", fee_bps, slippage_bps)
    cash_ledger, cash_summary = _apply_sleeves(frame, frame["cash_action"].eq("LONG"), "reference_price_proxy_gross_return", fee_bps, slippage_bps)
    _, doubled = _apply_sleeves(frame, "selected", "reference_price_proxy_gross_return", fee_bps * 2.0, slippage_bps * 2.0)
    for policy, policy_ledger in (("trend", trend_ledger), ("cash", cash_ledger)):
        frame[f"{policy}_selected"] = policy_ledger["selected"]
        frame[f"{policy}_sleeve_equity_before"] = policy_ledger["sleeve_equity_before"]
        frame[f"{policy}_sleeve_equity_after"] = policy_ledger["sleeve_equity_after"]
        frame[f"{policy}_sleeve_net_return"] = policy_ledger["sleeve_net_return"]
    primary = _augment_policy_summary(primary, frame, "selected")
    trend_summary = _augment_policy_summary(trend_summary, frame, "trend_selected")
    cash_summary = _augment_policy_summary(cash_summary, frame, "cash_selected")
    common = frame.loc[frame["common_valid_for_delay"]].copy()
    _, base_common = _apply_sleeves(common, "selected", "reference_price_proxy_gross_return", fee_bps, slippage_bps)
    delayed_frame = common.assign(reference_entry_utc=common["delayed_entry_utc"], reference_exit_utc=common["delayed_exit_utc"])
    _, delayed = _apply_sleeves(delayed_frame, "selected", "delayed_reference_price_proxy_gross_return", fee_bps, slippage_bps)
    comparison_counts = {
        "common_valid_count": len(common), "selected_count": int(common["selected"].sum()),
        "excluded_opportunity_count": int(frame["is_opportunity"].sum()) - len(common),
    }
    primary.update(
        {
            "benchmarks": {"cash": cash_summary, "trend": trend_summary},
            "doubled_costs_fixed_selections": doubled,
            "base_on_delay_common_subset": {**base_common, **comparison_counts},
            "delayed_reference_price_proxy": {**delayed, **comparison_counts, "target_id": "one_hour_delayed_open_to_open_plus_4h"},
        }
    )
    return frame, primary


def evaluate_forecast_arms(
    samples: pd.DataFrame, forecasts: dict[str, np.ndarray], config: dict
) -> dict[str, dict]:
    """Compare named forecast arms through the single-arm evaluator and metrics."""
    if not isinstance(forecasts, dict) or not forecasts:
        raise ValueError("forecasts must be a non-empty mapping of arm names to vectors")
    required = {"symbol", "target_valid", "target_return"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"samples missing required columns: {', '.join(missing)}")
    actual = pd.to_numeric(samples["target_return"], errors="raise").to_numpy(dtype=float)
    metric_rows = samples["target_valid"].eq(True).to_numpy() & np.isfinite(actual)
    strategy_summaries = {}
    forecast_results = {}
    ledgers = {}
    for name, values in forecasts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("forecast arm names must be non-empty strings")
        vector = np.asarray(values, dtype=float)
        ledger, strategy = evaluate_opportunities(samples, vector, config)
        strategy_summaries[name] = strategy
        ledgers[name] = ledger
        pooled_predictions = vector[metric_rows]
        pooled_actual = actual[metric_rows]
        per_asset = {}
        for symbol in sorted(samples.loc[metric_rows, "symbol"].unique()):
            asset_mask = metric_rows & samples["symbol"].eq(symbol).to_numpy()
            per_asset[symbol] = forecast_metrics(actual[asset_mask], vector[asset_mask])
        forecast_results[name] = {
            "pooled": forecast_metrics(pooled_actual, pooled_predictions),
            "per_asset": per_asset,
        }
    first_summary = next(iter(strategy_summaries.values()))
    strategy_summaries["cash"] = first_summary["benchmarks"]["cash"]
    strategy_summaries["trend"] = first_summary["benchmarks"]["trend"]
    first_ledger = ledgers[next(iter(forecasts))]
    ledgers["cash"] = _named_policy_ledger(first_ledger, "cash")
    ledgers["trend"] = _named_policy_ledger(first_ledger, "trend")
    passive_ledger, passive_summary = _buy_and_hold_reference(ledgers, config)
    ledgers["buy_and_hold"] = passive_ledger
    strategy_summaries["buy_and_hold"] = passive_summary
    policy_metrics = {
        name: {"pooled": _policy_metric_view(summary), "per_asset": summary["per_asset_metrics"]}
        for name, summary in strategy_summaries.items()
    }
    return {
        "strategy_summaries": strategy_summaries,
        "policy_metrics": policy_metrics,
        "forecast_metrics": forecast_results,
        "ledgers": ledgers,
    }


def _apply_sleeves(frame: pd.DataFrame, selected: str | pd.Series, gross_column: str, fee_bps: float, slippage_bps: float) -> tuple[pd.DataFrame, dict]:
    selected_mask = frame[selected].astype(bool) if isinstance(selected, str) else pd.Series(selected, index=frame.index).astype(bool)
    initial_equity = INITIAL_EQUITY
    symbols = sorted(frame["symbol"].unique())
    if set(symbols) - set(SYMBOL_MAPPING):
        raise ValueError("unsupported asset in opportunities")
    sleeve_initial = initial_equity / 2.0
    result = frame.copy()
    result["selected"] = selected_mask
    result["sleeve_equity_before"] = np.nan
    result["sleeve_equity_after"] = np.nan
    result["sleeve_net_return"] = np.nan
    asset_contributions: dict[str, dict[str, float | int]] = {}
    equity_by_asset = {symbol: sleeve_initial for symbol in SYMBOL_MAPPING}
    for symbol in symbols:
        index = result.index[result["symbol"].eq(symbol)]
        equity = sleeve_initial
        selected_count = 0
        for row_index in index:
            result.loc[row_index, "sleeve_equity_before"] = equity
            if selected_mask.loc[row_index]:
                gross = float(result.loc[row_index, gross_column])
                net = net_return(gross, fee_bps, slippage_bps)
                result.loc[row_index, "sleeve_net_return"] = net
                equity *= 1.0 + net
                selected_count += 1
            result.loc[row_index, "sleeve_equity_after"] = equity
        equity_by_asset[symbol] = equity
        asset_contributions[symbol] = {
            "selected_trades": selected_count,
            "starting_equity": sleeve_initial,
            "ending_equity": equity,
            "contribution": equity - sleeve_initial,
        }
    for symbol in SYMBOL_MAPPING:
        asset_contributions.setdefault(symbol, {"selected_trades": 0, "starting_equity": sleeve_initial, "ending_equity": sleeve_initial, "contribution": 0.0})
    per_asset_metrics = {}
    opportunities = result.loc[result["is_opportunity"]]
    span_start, span_end = opportunities["reference_entry_utc"].min(), opportunities["reference_exit_utc"].max()
    gross_equity = 0.0
    for symbol in SYMBOL_MAPPING:
        asset_rows = result["symbol"].eq(symbol)
        asset_selected = asset_rows & selected_mask
        asset_returns = result.loc[asset_selected, "sleeve_net_return"].dropna()
        asset_equity = asset_contributions[symbol]["ending_equity"]
        gross_return = _compound_return(result.loc[asset_selected, gross_column])
        gross_equity += sleeve_initial * (1.0 + (gross_return or 0.0))
        per_asset_metrics[symbol] = {
            "selected_trades": int(asset_selected.sum()),
            "average_net_return": float(asset_returns.mean()) if len(asset_returns) else None,
            "median_net_return": float(asset_returns.median()) if len(asset_returns) else None,
            "win_rate": float((asset_returns > 0.0).mean()) if len(asset_returns) else None,
            "cumulative_proxy_return": asset_equity / sleeve_initial - 1.0 if len(asset_returns) else None,
            "gross_cumulative_proxy_return": gross_return,
            "exposure": _capital_exposure(result.loc[asset_rows], span_start, span_end, sleeve_initial),
            "starting_equity": sleeve_initial,
            "ending_equity": asset_equity,
        }
    selected_returns = result.loc[selected_mask, "sleeve_net_return"].dropna()
    selected_count = int(selected_mask.sum())
    ending_equity = float(sum(equity_by_asset.values()))
    sampled_equity = _sampled_portfolio_equity(result, sleeve_initial)
    months = {}
    for month, rows in result.groupby(result["reference_entry_utc"].dt.strftime("%Y-%m"), sort=True):
        count = int(rows["selected"].sum())
        months[month] = {
            "selected_trades": count, "fraction_of_selected_trades": count / selected_count if selected_count else None,
            "net_pnl_usd": float((rows["sleeve_equity_after"] - rows["sleeve_equity_before"]).sum()),
        }
    summary: dict[str, Any] = {
        "status": "INSUFFICIENT_EVIDENCE" if selected_count == 0 else "REFERENCE_PRICE_PROXY_ONLY",
        "portfolio_starting_equity": initial_equity,
        "portfolio_ending_equity": ending_equity,
        "asset_contributions": asset_contributions,
        "per_asset_metrics": per_asset_metrics,
        "average_net_return": float(selected_returns.mean()) if selected_count else None,
        "median_net_return": float(selected_returns.median()) if selected_count else None,
        "win_rate": float((selected_returns > 0.0).mean()) if selected_count else None,
        "cumulative_proxy_return": ending_equity / initial_equity - 1.0 if selected_count else None,
        "gross_cumulative_proxy_return": gross_equity / initial_equity - 1.0 if selected_count else None,
        "sampled_drawdown": _max_drawdown(sampled_equity) if selected_count else None,
        "exposure": _capital_exposure(result, span_start, span_end, initial_equity),
        "temporal_concentration": {
            "months": months,
            "max_month_trade_fraction": max((row["selected_trades"] for row in months.values()), default=0) / selected_count if selected_count else None,
        },
    }
    return result, summary


def _capital_exposure(frame: pd.DataFrame, start, end, initial_equity: float) -> dict:
    """USD-hours with equity marked only at realized exits; idle/gap time stays cash."""
    selected = frame.loc[frame["selected"]]
    span_hours = float((end - start) / pd.Timedelta(hours=1)) if pd.notna(start) and pd.notna(end) else 0.0
    holding_hours = (selected["reference_exit_utc"] - selected["reference_entry_utc"]) / pd.Timedelta(hours=1)
    invested = float((selected["sleeve_equity_before"] * holding_hours).sum())
    remaining_hours = (end - selected["reference_exit_utc"]) / pd.Timedelta(hours=1)
    pnl = selected["sleeve_equity_after"] - selected["sleeve_equity_before"]
    available = initial_equity * span_hours + float((pnl * remaining_hours).sum())
    return {
        "selected_trade_count": len(selected), "selected_hours": float(holding_hours.sum()),
        "fraction_of_rows": len(selected) / len(frame) if len(frame) else None,
        "span_start_utc": start.isoformat() if pd.notna(start) else None,
        "span_end_utc": end.isoformat() if pd.notna(end) else None,
        "available_asset_hours": span_hours * initial_equity / (INITIAL_EQUITY / 2),
        "capital_time_usd_hours": invested, "available_capital_time_usd_hours": available,
        "capital_time_fraction": invested / available if available > 0.0 else None,
        "basis": "USD-hours; sleeve capital marked at exits, including cash time between opportunities",
    }


def _buy_and_hold_reference(ledgers: dict[str, pd.DataFrame], config: dict) -> tuple[pd.DataFrame, dict]:
    """Use the longest common contiguous opportunity span, with earliest tie breaking."""
    first = next(iter(ledgers.values()))
    eligible = first.loc[first["is_opportunity"]]
    counts = eligible.groupby("reference_entry_utc")["symbol"].nunique()
    times = counts.index[counts.eq(len(SYMBOL_MAPPING))].to_series()
    blocks = [block for _, block in times.groupby(times.diff().ne(pd.Timedelta(hours=4)).cumsum())]
    span = max(blocks, key=len) if blocks else times
    # Only observed, valid four-hour windows define the span; never bridge a gap.
    passive = eligible.iloc[:0][["symbol", "reference_entry_utc", "reference_exit_utc", "reference_entry_price", "reference_exit_price"]].copy()
    start = end = None
    if len(span):
        start, last_entry = span.iloc[0], span.iloc[-1]
        entry_rows = eligible.loc[eligible["reference_entry_utc"].eq(start)].set_index("symbol")
        exit_rows = eligible.loc[eligible["reference_entry_utc"].eq(last_entry)].set_index("symbol")
        passive = entry_rows[["reference_entry_utc", "reference_entry_price"]].copy()
        passive["reference_exit_utc"] = exit_rows["reference_exit_utc"]
        passive["reference_exit_price"] = exit_rows["reference_exit_price"]
        passive = passive.reset_index()
        end = passive["reference_exit_utc"].max()
    passive["is_opportunity"] = True
    passive["selected"] = True
    passive["action"] = "LONG"
    passive["forecast_gross_return"] = np.nan
    passive["reference_price_proxy_gross_return"] = passive["reference_exit_price"] / passive["reference_entry_price"] - 1
    passive, summary = _apply_sleeves(passive, "selected", "reference_price_proxy_gross_return", *_costs(config))
    passive["reference_price_proxy_net_return"] = passive["sleeve_net_return"]
    summary = _augment_policy_summary(summary, passive, "selected")
    same_span = {}
    for name, ledger in ledgers.items():
        rows = ledger.loc[ledger["selected"] & ledger["reference_entry_utc"].isin(span)]
        same_span[name] = {
            "selected_trades": len(rows),
            "net_cumulative_proxy_return": sum(
                _compound_return(rows.loc[rows["symbol"].eq(symbol), "sleeve_net_return"]) or 0.0
                for symbol in SYMBOL_MAPPING
            ) / 2.0,
        }
    summary.update({
        "span_start_utc": start.isoformat().replace("+00:00", "Z") if start is not None else None,
        "span_end_utc": end.isoformat().replace("+00:00", "Z") if end is not None else None,
        "span_selection": "longest common contiguous valid opportunity span; earliest tie",
        "common_opportunities_in_span_per_asset": len(span),
        "excluded_common_opportunities_per_asset": len(times) - len(span),
        "same_span_policy_returns": same_span,
        "exposure_caveat": "Buy-and-hold invests both sleeves continuously over this span, with one entry and one exit cost per asset. Scheduled policies can hold cash and pay costs on every selected block. Endpoint-only passive sampled drawdown misses all intervening losses.",
    })
    return passive, summary


def _named_policy_ledger(ledger: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Project a shared opportunity ledger into one named policy's action/equity view."""
    if policy not in {"cash", "trend"}:
        raise ValueError("named policy ledger must be cash or trend")
    result = ledger.copy()
    result["policy_name"] = policy
    result["action"] = result[f"{policy}_action"]
    result["selected"] = result[f"{policy}_selected"]
    result["sleeve_equity_before"] = result[f"{policy}_sleeve_equity_before"]
    result["sleeve_equity_after"] = result[f"{policy}_sleeve_equity_after"]
    result["sleeve_net_return"] = result[f"{policy}_sleeve_net_return"]
    result["reference_price_proxy_net_return"] = result["sleeve_net_return"]
    result["delayed_reference_price_proxy_net_return"] = np.nan  # no benchmark delay stress is reported
    result["forecast_gross_return"] = np.nan
    return result


def _policy_metric_view(summary: dict) -> dict:
    """Keep policy performance metrics separate from forecast metrics."""
    return {
        key: summary[key]
        for key in (
            "historical_return_basis", "selected_opportunities", "all_opportunities", "coverage",
            "status", "average_net_return", "median_net_return", "win_rate",
            "cumulative_proxy_return", "gross_cumulative_proxy_return", "sampled_drawdown", "exposure", "temporal_concentration",
        )
    }


def _augment_policy_summary(summary: dict, frame: pd.DataFrame, selected: str | pd.Series) -> dict:
    """Add shared opportunity fields to every policy's JSON-safe summary."""
    selected_mask = frame[selected].astype(bool) if isinstance(selected, str) else pd.Series(selected, index=frame.index).astype(bool)
    common_opportunities = frame["is_opportunity"].astype(bool)
    selected_count = int((selected_mask & common_opportunities).sum())
    common_count = int(common_opportunities.sum())
    augmented = dict(summary)
    augmented.update(
        {
            "historical_return_basis": "reference_price_proxy",
            "selected_opportunities": selected_count,
            "all_opportunities": common_count,
            "coverage": float(selected_count / common_count) if common_count else None,
        }
    )
    return augmented


def _nonoverlapping_selection(frame: pd.DataFrame, candidate: pd.Series) -> pd.Series:
    """Keep the first chronological LONG in each asset sleeve until its exit boundary."""
    selected = pd.Series(False, index=frame.index, dtype=bool)
    for _, group in frame.groupby("symbol", sort=False):
        next_available = pd.Timestamp.min.tz_localize("UTC")
        for row_index, row in group.iterrows():
            if candidate.loc[row_index] and row["reference_entry_utc"] >= next_available:
                selected.loc[row_index] = True
                next_available = row["reference_exit_utc"]
    return selected


def _sampled_portfolio_equity(frame: pd.DataFrame, sleeve_initial: float) -> list[float]:
    states = {symbol: sleeve_initial for symbol in SYMBOL_MAPPING}
    values = [sum(states.values())]
    for _, group in frame.groupby("reference_entry_utc", sort=True):
        for _, row in group.iterrows():
            states[row["symbol"]] = float(row["sleeve_equity_after"])
        values.append(sum(states.values()))
    return values


def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0]
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    return float(drawdown)


def _net_series(gross: pd.Series, selected: pd.Series, fee_bps: float, slippage_bps: float) -> pd.Series:
    result = pd.Series(np.nan, index=gross.index, dtype=float)
    result.loc[selected] = [net_return(float(value), fee_bps, slippage_bps) for value in gross.loc[selected]]
    return result


def _compound_return(returns: pd.Series) -> float | None:
    if len(returns) == 0:
        return None
    return float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0)


def _costs(config: dict) -> tuple[float, float]:
    try:
        fee_bps = float(config["fee_bps_per_side"])
        slippage_bps = float(config["slippage_bps_per_side"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("config must specify per-side fee and slippage costs") from error
    net_return(0.0, fee_bps, slippage_bps)
    return fee_bps, slippage_bps


def _validate_policy_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("policy config must be a mapping")
    try:
        hours = tuple(config.get("opportunity_hours_utc", OPPORTUNITY_HOURS))
    except TypeError as error:
        raise ValueError("fixed opportunity hours must be 00,04,08,12,16,20 UTC") from error
    if hours != OPPORTUNITY_HOURS:
        raise ValueError("fixed opportunity hours must be 00,04,08,12,16,20 UTC")
    try:
        initial_equity = float(config.get("initial_equity", INITIAL_EQUITY))
    except (TypeError, ValueError) as error:
        raise ValueError("fixed initial equity must be $10,000") from error
    if initial_equity != INITIAL_EQUITY:
        raise ValueError("fixed initial equity must be $10,000")
    try:
        threshold = float(config.get("minimum_predicted_net_return", MINIMUM_PREDICTED_NET_RETURN))
    except (TypeError, ValueError) as error:
        raise ValueError("fixed minimum predicted net return must be 0.001") from error
    if threshold != MINIMUM_PREDICTED_NET_RETURN:
        raise ValueError("fixed minimum predicted net return must be 0.001")


def _utc_config(config: dict, name: str) -> pd.Timestamp:
    if name not in config:
        raise ValueError(f"config missing {name}")
    value = pd.Timestamp(config[name])
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value.tz_convert("UTC")


def _utc_series(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if not isinstance(parsed, pd.Series):
        parsed = pd.Series(parsed, index=values.index)
    if parsed.dt.tz is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return parsed.dt.tz_convert("UTC")


def _validate_metadata(metadata: dict) -> None:
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise ModelSchemaError("metadata keys are incomplete or unexpected")
    features = metadata["feature_columns"]
    if features != FEATURE_COLUMNS:
        raise ModelSchemaError("feature columns must exactly match FEATURE_COLUMNS in order")
    if metadata["symbol_mapping"] != SYMBOL_MAPPING:
        raise ModelSchemaError("symbol mapping must exactly define BTCUSDT and ETHUSDT")
    if metadata["target_id"] != TARGET_ID:
        raise ModelSchemaError("target identifier does not match the fixed target")
    try:
        _utc_config({"training_cutoff_utc": metadata["training_cutoff_utc"]}, "training_cutoff_utc")
    except (TypeError, ValueError) as error:
        raise ModelSchemaError("training cutoff must be timezone-aware UTC") from error
    if metadata["seed"] != 42 or not isinstance(metadata["seed"], int) or isinstance(metadata["seed"], bool):
        raise ModelSchemaError("seed must exactly equal 42")
    if not isinstance(metadata["estimator_settings"], dict) or metadata["estimator_settings"] != TREE_SETTINGS:
        raise ModelSchemaError("tree estimator settings do not match the fixed arm")
    versions = metadata["versions"]
    if not isinstance(versions, dict) or set(versions) != _VERSION_KEYS or any(not isinstance(value, str) or not value for value in versions.values()):
        raise ModelSchemaError("package versions are incomplete")
    hashes = metadata["hashes"]
    if not isinstance(hashes, dict) or set(hashes) != _HASH_KEYS or any(not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()) for value in hashes.values()):
        raise ModelSchemaError("hashes must be SHA-256 hex strings")


def _validate_tree_settings(model, expected: dict) -> None:
    parameters = model.get_params()
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ModelSchemaError(f"native tree setting mismatch: {key}")
