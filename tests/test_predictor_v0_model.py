import json

import numpy as np
import pandas as pd
import pytest

from experiments.predictor_v0.data import FEATURE_COLUMNS
from experiments.predictor_v0.model import (
    ModelSchemaError,
    TREE_SETTINGS,
    choose_action,
    evaluate_opportunities,
    evaluate_forecast_arms,
    fit_models,
    forecast_metrics,
    load_tree,
    net_return,
    save_tree,
    split_samples,
)


def config():
    return {
        "sample_start_utc": "2025-01-01T00:00:00Z",
        "train_end_utc": "2025-01-03T00:00:00Z",
        "development_end_utc": "2025-01-05T00:00:00Z",
        "holdout_end_utc": "2025-01-07T00:00:00Z",
        "fee_bps_per_side": 10.0,
        "slippage_bps_per_side": 5.0,
        "minimum_predicted_net_return": 0.001,
        "opportunity_hours_utc": [0, 4, 8, 12, 16, 20],
        "initial_equity": 10_000.0,
        "seed": 42,
        "max_threads": 4,
    }


def sample_frame(periods=144):
    times = pd.date_range("2025-01-01T00:00:00Z", periods=periods, freq="h")
    rows = []
    for symbol, offset in (("BTCUSDT", 0.0), ("ETHUSDT", 1.0)):
        for index, cutoff in enumerate(times):
            entry = 100.0 + offset + index
            target = 0.004 if index % 3 else -0.002
            delayed_target = target - 0.0005
            row = {
                "symbol": symbol,
                "feature_cutoff_utc": cutoff,
                "label_end_utc": cutoff + pd.Timedelta(hours=4),
                "reference_entry_utc": cutoff,
                "reference_exit_utc": cutoff + pd.Timedelta(hours=4),
                "reference_entry_price": entry,
                "reference_exit_price": entry * (1 + target),
                "delayed_entry_utc": cutoff + pd.Timedelta(hours=1),
                "delayed_exit_utc": cutoff + pd.Timedelta(hours=5),
                "delayed_entry_price": entry * 1.001,
                "delayed_exit_price": entry * 1.001 * (1 + delayed_target),
                "target_return": target,
                "target_valid": True,
            }
            for feature_index, feature in enumerate(FEATURE_COLUMNS):
                row[feature] = (index + 1) * (feature_index + 1) / 10_000 + offset
            row["is_eth"] = 1.0 if symbol == "ETHUSDT" else 0.0
            row["price_vs_sma20"] = 0.01
            row["return_24h"] = 0.01
            rows.append(row)
    return pd.DataFrame(rows)


def tree_metadata():
    return {
        "feature_columns": FEATURE_COLUMNS,
        "symbol_mapping": {"BTCUSDT": 0, "ETHUSDT": 1},
        "target_id": "next_hour_open_to_open_plus_4h",
        "training_cutoff_utc": "2025-01-03T00:00:00Z",
        "seed": 42,
        "estimator_settings": {
            "objective": "reg:squarederror", "tree_method": "hist", "device": "cpu",
            "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
            "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.0, "reg_lambda": 10.0, "random_state": 42, "n_jobs": 4,
        },
        "versions": {"numpy": "2.4.6", "pandas": "3.0.5", "scikit_learn": "1.9.0", "xgboost": "3.2.0"},
        "hashes": {key: "a" * 64 for key in ("data", "config", "code", "feature_schema")},
    }


def test_split_is_purged_chronological_and_keeps_both_assets_together():
    samples = sample_frame()
    splits = split_samples(samples.sample(frac=1.0, random_state=7), config())

    assert set(splits) == {"train", "development", "holdout"}
    assert set(splits["development"]["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert (splits["train"]["feature_cutoff_utc"] < pd.Timestamp("2025-01-03T00:00:00Z")).all()
    assert (splits["train"]["label_end_utc"] < pd.Timestamp("2025-01-03T00:00:00Z")).all()
    assert splits["train"].sort_values(["feature_cutoff_utc", "symbol"]).index.tolist() == splits["train"].index.tolist()


def test_fixed_models_reject_invalid_input_and_future_rows_do_not_change_ridge_scaler():
    samples = sample_frame()
    train = split_samples(samples, config())["train"]
    models = fit_models(train, FEATURE_COLUMNS, config())
    changed_future = samples.copy()
    changed_future.loc[changed_future["feature_cutoff_utc"] >= pd.Timestamp("2025-01-05T00:00:00Z"), FEATURE_COLUMNS] = 9_999.0
    expanded_models = fit_models(split_samples(changed_future, config())["train"], FEATURE_COLUMNS, config())

    assert set(models) == {"zero", "ridge", "tree"}
    assert models["ridge"].named_steps["ridge"].alpha == 1.0
    assert models["tree"].get_params()["n_estimators"] == 300
    np.testing.assert_allclose(models["ridge"].named_steps["standardscaler"].mean_, expanded_models["ridge"].named_steps["standardscaler"].mean_)
    invalid = train.copy()
    invalid.loc[invalid.index[0], FEATURE_COLUMNS[0]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_models(invalid, FEATURE_COLUMNS, config())


def test_forecast_metrics_returns_defined_and_undefined_statistics():
    metrics = forecast_metrics(np.array([0.01, -0.01, 0.02]), np.array([0.02, -0.03, 0.0]))
    undefined = forecast_metrics(np.array([0.0, 0.0]), np.array([0.0, 0.0]))

    assert metrics["count"] == 3
    assert metrics["pearson_correlation"] == pytest.approx(0.737043474095502)
    assert metrics["oos_r2_vs_zero"] is not None
    assert metrics["direction_agreement"] == pytest.approx(2 / 3)
    assert undefined["pearson_correlation"] is None
    assert undefined["pearson_reason"] == "ZERO_VARIANCE"
    assert undefined["oos_r2_vs_zero"] is None
    assert undefined["oos_r2_reason"] == "ZERO_TARGET_ENERGY"


def test_native_tree_round_trip_requires_exact_metadata(tmp_path):
    train = split_samples(sample_frame(), config())["train"]
    tree = fit_models(train, FEATURE_COLUMNS, config())["tree"]
    model_path = tmp_path / "tree.json"
    metadata = tree_metadata()
    save_tree(tree, model_path, metadata)
    restored = load_tree(model_path, model_path.with_suffix(".metadata.json"))

    np.testing.assert_allclose(tree.predict(train[FEATURE_COLUMNS]), restored.predict(train[FEATURE_COLUMNS]))
    assert {key: restored.get_params()[key] for key in TREE_SETTINGS} == TREE_SETTINGS
    native_settings = json.loads(restored.get_booster().save_config())["learner"]["generic_param"]
    assert int(native_settings["nthread"]) == 4
    mismatched = json.loads(json.dumps(metadata))
    mismatched["feature_columns"] = list(reversed(FEATURE_COLUMNS))
    model_path.with_suffix(".metadata.json").write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ModelSchemaError, match="MODEL_SCHEMA_ERROR"):
        load_tree(model_path, model_path.with_suffix(".metadata.json"))
    mismatched["feature_columns"] = FEATURE_COLUMNS
    mismatched["seed"] = 7
    model_path.with_suffix(".metadata.json").write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ModelSchemaError, match="MODEL_SCHEMA_ERROR"):
        load_tree(model_path, model_path.with_suffix(".metadata.json"))
    model_path.with_suffix(".metadata.json").write_text(json.dumps({"feature_columns": FEATURE_COLUMNS}), encoding="utf-8")
    with pytest.raises(ModelSchemaError, match="MODEL_SCHEMA_ERROR"):
        load_tree(model_path, model_path.with_suffix(".metadata.json"))


def test_cost_formula_and_action_threshold_are_strict():
    expected = (1.01 * 0.9995 * 0.999) / (1.0005 * 1.001) - 1.0
    assert net_return(0.01, 10.0, 5.0) == pytest.approx(expected)
    assert choose_action(0.003, config()) == "NO_TRADE"
    assert choose_action(0.0041, config()) == "LONG"
    nonstandard_config = config()
    nonstandard_config["minimum_predicted_net_return"] = 0.005
    with pytest.raises(ValueError, match="fixed minimum"):
        choose_action(0.0041, nonstandard_config)
    with pytest.raises(ValueError, match="gross"):
        net_return(-1.0, 10.0, 5.0)
    with pytest.raises(ValueError, match="cost"):
        net_return(0.01, -1.0, 5.0)
    with pytest.raises(ValueError, match="cost"):
        net_return(0.01, 10_000.0, 5.0)


def test_evaluation_reconciles_independent_sleeves_and_delayed_references():
    samples = sample_frame(periods=12)
    forecasts = np.full(len(samples), 0.01)
    ledger, summary = evaluate_opportunities(samples, forecasts, config())

    selected = ledger.loc[ledger["selected"]]
    assert set(selected["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert (selected["reference_price_proxy_net_return"] < selected["reference_price_proxy_gross_return"]).all()
    assert (selected["delayed_entry_utc"] == selected["reference_entry_utc"] + pd.Timedelta(hours=1)).all()
    assert (selected["delayed_exit_utc"] == selected["reference_exit_utc"] + pd.Timedelta(hours=1)).all()
    np.testing.assert_allclose(
        selected["delayed_reference_price_proxy_gross_return"],
        selected["delayed_exit_price"] / selected["delayed_entry_price"] - 1.0,
    )
    assert summary["historical_return_basis"] == "reference_price_proxy"
    assert summary["portfolio_ending_equity"] == pytest.approx(sum(value["ending_equity"] for value in summary["asset_contributions"].values()))
    assert summary["delayed_reference_price_proxy"]["common_valid_count"] == len(ledger.loc[ledger["common_valid_for_delay"]])
    assert summary["doubled_costs_fixed_selections"]["cumulative_proxy_return"] < summary["cumulative_proxy_return"]
    for _, asset_rows in selected.groupby("symbol"):
        np.testing.assert_allclose(
            asset_rows["sleeve_equity_before"].iloc[1:].to_numpy(),
            asset_rows["sleeve_equity_after"].iloc[:-1].to_numpy(),
        )

    _, zero_summary = evaluate_opportunities(samples, np.zeros(len(samples)), config())
    assert summary["all_opportunities"] == zero_summary["all_opportunities"]


def test_evaluation_rejects_bad_reference_order_and_marks_zero_trades_insufficient():
    samples = sample_frame(periods=8)
    bad = samples.copy()
    bad.loc[bad.index[0], "reference_exit_utc"] = bad.loc[bad.index[0], "reference_entry_utc"]
    with pytest.raises(ValueError, match="reference offsets"):
        evaluate_opportunities(bad, np.zeros(len(bad)), config())

    ledger, summary = evaluate_opportunities(samples, np.zeros(len(samples)), config())
    assert not ledger["selected"].any()
    assert summary["status"] == "INSUFFICIENT_EVIDENCE"
    assert summary["average_net_return"] is None
    assert summary["win_rate"] is None


def test_evaluation_never_overlaps_a_sleeve_when_opportunities_are_dense():
    dense_config = config()
    dense_config["opportunity_hours_utc"] = list(range(24))
    with pytest.raises(ValueError, match="fixed opportunity hours"):
        evaluate_opportunities(sample_frame(periods=12), np.full(24, 0.01), dense_config)


def test_unordered_mixed_validity_stays_attached_to_each_row():
    samples = sample_frame(periods=8)
    samples.loc[0, "target_valid"] = False
    samples.loc[1, "delayed_exit_price"] = np.nan
    shuffled = samples.sample(frac=1.0, random_state=11).reset_index(drop=True)

    ledger, _ = evaluate_opportunities(shuffled, np.full(len(shuffled), 0.01), config())
    invalid_reference = ledger.loc[ledger["feature_cutoff_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")].iloc[0]
    invalid_delay = ledger.loc[ledger["feature_cutoff_utc"] == pd.Timestamp("2025-01-01T01:00:00Z")].iloc[0]
    assert not invalid_reference["is_opportunity"]
    assert pd.isna(invalid_reference["reference_price_proxy_gross_return"])
    assert not invalid_delay["common_valid_for_delay"]
    assert pd.isna(invalid_delay["delayed_reference_price_proxy_gross_return"])


def test_shuffled_delayed_validity_stays_attached_to_proxy_and_net_return():
    samples = sample_frame(periods=12)
    samples.loc[samples["feature_cutoff_utc"] == pd.Timestamp("2025-01-01T00:00:00Z"), ["delayed_entry_price", "delayed_exit_price"]] = np.nan
    shuffled = samples.iloc[::-1].reset_index(drop=True)

    ledger, _ = evaluate_opportunities(shuffled, np.full(len(shuffled), 0.01), config())

    invalid = ledger.loc[ledger["feature_cutoff_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")].iloc[0]
    valid = ledger.loc[ledger["feature_cutoff_utc"] == pd.Timestamp("2025-01-01T04:00:00Z")].iloc[0]
    assert not invalid["delayed_valid"]
    assert pd.isna(invalid["delayed_reference_price_proxy_gross_return"])
    assert pd.isna(invalid["delayed_reference_price_proxy_net_return"])
    assert valid["delayed_valid"]
    assert valid["delayed_reference_price_proxy_net_return"] < 0.0035


def test_missing_base_reference_timestamps_make_proxy_invalid():
    samples = sample_frame(periods=8)
    missing = samples["feature_cutoff_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")
    samples.loc[missing, ["reference_entry_utc", "reference_exit_utc"]] = pd.NaT

    ledger, _ = evaluate_opportunities(samples, np.zeros(len(samples)), config())
    row = ledger.loc[ledger["feature_cutoff_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")].iloc[0]
    assert not row["valid_reference"]
    assert not row["is_opportunity"]
    assert pd.isna(row["reference_price_proxy_gross_return"])


def test_fixed_policy_rejects_custom_schedules_and_sleeve_equity():
    bad_schedule = config()
    bad_schedule["opportunity_hours_utc"] = [0, 4, 8]
    with pytest.raises(ValueError, match="fixed opportunity hours"):
        evaluate_opportunities(sample_frame(periods=12), np.full(24, 0.01), bad_schedule)
    bad_equity = config()
    bad_equity["initial_equity"] = 9_999.0
    with pytest.raises(ValueError, match="fixed initial equity"):
        evaluate_opportunities(sample_frame(periods=12), np.full(24, 0.01), bad_equity)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.assign(reference_exit_utc=frame["reference_entry_utc"] + pd.Timedelta(hours=3)),
        lambda frame: frame.assign(delayed_entry_utc=frame["reference_entry_utc"] + pd.Timedelta(hours=2)),
        lambda frame: frame.assign(delayed_exit_utc=frame["reference_exit_utc"] + pd.Timedelta(hours=2)),
    ],
)
def test_evaluation_rejects_non_four_hour_or_one_hour_reference_offsets(mutator):
    with pytest.raises(ValueError, match="reference offsets"):
        evaluate_opportunities(mutator(sample_frame(periods=12)), np.full(24, 0.01), config())


def test_terminal_delayed_rows_are_excluded_and_arm_comparison_reports_forecast_metrics():
    samples = sample_frame(periods=120)
    terminal = samples["feature_cutoff_utc"] == pd.Timestamp("2025-01-01T11:00:00Z")
    samples.loc[terminal, ["delayed_entry_utc", "delayed_exit_utc", "delayed_entry_price", "delayed_exit_price"]] = np.nan
    forecasts = {"tree": np.full(len(samples), 0.01), "ridge": np.zeros(len(samples)), "zero": np.zeros(len(samples))}

    comparison = evaluate_forecast_arms(samples, forecasts, config())

    assert set(comparison["strategy_summaries"]) == {"tree", "ridge", "zero", "cash", "trend", "buy_and_hold"}
    assert set(comparison["ledgers"]) == {"tree", "ridge", "zero", "cash", "trend", "buy_and_hold"}
    assert set(comparison["policy_metrics"]) == {"tree", "ridge", "zero", "cash", "trend", "buy_and_hold"}
    assert set(comparison["forecast_metrics"]) == {"tree", "ridge", "zero"}
    assert comparison["forecast_metrics"]["tree"]["pooled"]["count"] == len(samples)
    assert comparison["forecast_metrics"]["tree"]["per_asset"]["BTCUSDT"]["count"] == 120
    assert comparison["policy_metrics"]["cash"]["per_asset"]["BTCUSDT"]["selected_trades"] == 0
    assert comparison["policy_metrics"]["trend"]["per_asset"]["ETHUSDT"]["selected_trades"] > 0
    common_opportunities = int(comparison["ledgers"]["tree"]["is_opportunity"].sum())
    primary_summary = comparison["strategy_summaries"]["tree"]
    assert primary_summary["selected_opportunities"] == int(comparison["ledgers"]["tree"]["selected"].sum())
    assert primary_summary["coverage"] == pytest.approx(1.0)
    assert comparison["ledgers"]["tree"]["forecast_gross_return"].notna().all()
    for policy in ("cash", "trend"):
        summary = comparison["strategy_summaries"][policy]
        pooled = comparison["policy_metrics"][policy]["pooled"]
        assert summary["historical_return_basis"] == "reference_price_proxy"
        assert summary["all_opportunities"] == common_opportunities
        assert summary["selected_opportunities"] == pooled["selected_opportunities"]
        assert summary["coverage"] == pooled["coverage"]
        assert comparison["ledgers"][policy]["forecast_gross_return"].isna().all()
    assert comparison["strategy_summaries"]["cash"]["selected_opportunities"] == 0
    assert comparison["strategy_summaries"]["cash"]["coverage"] == 0.0
    assert not comparison["ledgers"]["cash"]["selected"].any()
    trend_trade_count = sum(
        metrics["selected_trades"] for metrics in comparison["policy_metrics"]["trend"]["per_asset"].values()
    )
    assert comparison["strategy_summaries"]["trend"]["selected_opportunities"] == trend_trade_count
    assert comparison["strategy_summaries"]["trend"]["coverage"] == pytest.approx(trend_trade_count / common_opportunities)
    assert not comparison["ledgers"]["tree"].loc[
        comparison["ledgers"]["tree"]["feature_cutoff_utc"] == pd.Timestamp("2025-01-04T20:00:00Z"),
        "common_valid_for_delay",
    ].any()


def test_policy_projection_has_only_its_own_net_returns():
    samples = sample_frame(periods=12)
    samples.loc[samples["feature_cutoff_utc"].dt.hour == 4, "price_vs_sma20"] = -1.0
    forecast = np.where(samples["feature_cutoff_utc"].dt.hour == 4, 0.01, 0.0)
    comparison = evaluate_forecast_arms(samples, {"tree": forecast}, config())
    for policy in ("cash", "trend"):
        ledger = comparison["ledgers"][policy]
        assert ledger["forecast_gross_return"].isna().all()
        assert ledger.loc[~ledger["selected"], "reference_price_proxy_net_return"].isna().all()
        pd.testing.assert_series_equal(ledger["reference_price_proxy_net_return"], ledger["sleeve_net_return"], check_names=False)
        assert ledger["delayed_reference_price_proxy_net_return"].isna().all()
        selected = ledger.loc[ledger["selected"]]
        assert selected["reference_price_proxy_net_return"].tolist() == pytest.approx([
            net_return(gross, 10.0, 5.0) for gross in selected["reference_price_proxy_gross_return"]
        ])


def test_delay_and_gross_returns_use_independent_sleeves_and_common_subset():
    samples = sample_frame(periods=8)
    btc = samples["symbol"].eq("BTCUSDT")
    samples["delayed_exit_price"] = samples["delayed_entry_price"] * np.where(btc, 1.1, 0.9)
    samples.loc[0, ["delayed_exit_utc", "delayed_exit_price"]] = [pd.NaT, np.nan]
    ledger, summary = evaluate_opportunities(samples, np.full(len(samples), 0.01), config())
    delayed = summary["delayed_reference_price_proxy"]
    expected_sleeves = [5000 * (1 + net_return(0.1, 10, 5)), 5000 * (1 + net_return(-0.1, 10, 5))**2]
    assert delayed["portfolio_ending_equity"] == pytest.approx(sum(expected_sleeves))
    assert delayed["cumulative_proxy_return"] == pytest.approx(sum(expected_sleeves) / 10000 - 1)
    assert delayed["selected_count"] == 3
    assert delayed["excluded_opportunity_count"] == 1
    assert summary["base_on_delay_common_subset"]["selected_count"] == 3
    expected_gross = sum(5000 * np.prod(1 + group.loc[group["selected"], "reference_price_proxy_gross_return"]) for _, group in ledger.groupby("symbol")) / 10000 - 1
    assert summary["gross_cumulative_proxy_return"] == pytest.approx(expected_gross)


def test_capital_time_exposure_and_monthly_concentration_have_explicit_units():
    samples = sample_frame(periods=8)
    forecasts = np.where(samples["symbol"].eq("BTCUSDT") & samples["feature_cutoff_utc"].dt.hour.eq(0), 0.01, 0.0)
    ledger, summary = evaluate_opportunities(samples, forecasts, config())
    exposure = summary["exposure"]
    pnl = ledger.loc[ledger["selected"], "sleeve_equity_after"].iloc[0] - 5000
    assert exposure["capital_time_usd_hours"] == 5000 * 4
    assert exposure["available_capital_time_usd_hours"] == pytest.approx(10000 * 8 + pnl * 4)
    assert exposure["capital_time_fraction"] == pytest.approx(20000 / (80000 + pnl * 4))
    assert exposure["selected_hours"] == 4
    assert exposure["available_asset_hours"] == 16
    concentration = summary["temporal_concentration"]
    assert concentration["months"]["2025-01"]["selected_trades"] == 1
    assert concentration["months"]["2025-01"]["net_pnl_usd"] == pytest.approx(pnl)
    assert concentration["max_month_trade_fraction"] == 1.0


def test_buy_and_hold_uses_longest_common_contiguous_span_with_costs_and_no_forecast():
    samples = sample_frame(periods=28)
    # A missing ETH opportunity separates [00,08) from the longer [12,28) span.
    samples.loc[samples["symbol"].eq("ETHUSDT") & samples["feature_cutoff_utc"].dt.hour.eq(8), "target_valid"] = False
    comparison = evaluate_forecast_arms(samples, {"tree": np.full(len(samples), 0.01)}, config())
    passive = comparison["strategy_summaries"]["buy_and_hold"]
    ledger = comparison["ledgers"]["buy_and_hold"]
    assert passive["span_start_utc"] == "2025-01-01T12:00:00Z"
    assert passive["span_end_utc"] == "2025-01-02T04:00:00Z"
    assert "continuous" in passive["exposure_caveat"].lower()
    assert passive["exposure"]["capital_time_fraction"] == 1.0
    assert ledger["forecast_gross_return"].isna().all()
    expected = sum(5000 * (1 + net_return(row.reference_exit_price / row.reference_entry_price - 1, 10, 5)) for row in ledger.itertuples())
    assert passive["portfolio_ending_equity"] == pytest.approx(expected)
    assert passive["same_span_policy_returns"]["cash"]["net_cumulative_proxy_return"] == 0.0
    assert set(comparison["forecast_metrics"]) == {"tree"}
