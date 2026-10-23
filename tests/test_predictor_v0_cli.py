import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.predictor_v0 import run


def _config() -> dict:
    return {
        "experiment_id": "crypto_4h_tree_v0",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "venue": "binance_spot",
        "interval": "1h",
        "raw_start_utc": "2025-01-01T00:00:00Z",
        "sample_start_utc": "2025-01-03T12:00:00Z",
        "train_end_utc": "2025-01-05T12:00:00Z",
        "development_end_utc": "2025-01-07T00:00:00Z",
        "holdout_end_utc": "2025-01-08T00:00:00Z",
        "horizon_hours": 4,
        "fee_bps_per_side": 10.0,
        "slippage_bps_per_side": 5.0,
        "minimum_predicted_net_return": 0.001,
        "opportunity_hours_utc": [0, 4, 8, 12, 16, 20],
        "seed": 42,
        "max_threads": 4,
        "initial_equity": 10000.0,
        "mode": "paper_only",
    }


def _candles(periods: int = 180) -> pd.DataFrame:
    times = pd.date_range("2025-01-01T00:00:00Z", periods=periods, freq="h")
    frames = []
    for symbol, base in (("BTCUSDT", 100.0), ("ETHUSDT", 200.0)):
        trend = np.arange(periods, dtype=float)
        open_ = base + trend + 0.05 * np.sin(trend / 3.0)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "open_time_utc": times,
                    "close_time_utc": times + pd.Timedelta(hours=1),
                    "open": open_,
                    "high": open_ + 1.0,
                    "low": open_ - 1.0,
                    "close": open_ + 0.25,
                    "volume": 1000.0 + trend,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _archive_payload(candles: pd.DataFrame, symbol: str) -> bytes:
    rows = []
    for row in candles.loc[candles["symbol"].eq(symbol)].itertuples(index=False):
        open_us = int(pd.Timestamp(row.open_time_utc).value // 1_000)
        close_us = int((pd.Timestamp(row.close_time_utc) - pd.Timedelta(microseconds=1)).value // 1_000)
        rows.append([open_us, row.open, row.high, row.low, row.close, row.volume, close_us, 0, 0, 0, 0, 0])
    raw = "\n".join(",".join(str(value) for value in row) for row in rows).encode("utf-8")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-1h-2025-01.csv", raw)
    return stream.getvalue()


def _archive_transport(candles: pd.DataFrame):
    payloads = {symbol: _archive_payload(candles, symbol) for symbol in ("BTCUSDT", "ETHUSDT")}

    def transport(url: str) -> bytes:
        symbol = next(symbol for symbol in payloads if f"/{symbol}/" in url)
        payload = payloads[symbol]
        if url.endswith(".CHECKSUM"):
            return f"{hashlib.sha256(payload).hexdigest()} {symbol}.zip".encode("ascii")
        return payload

    return transport


def _api_payload(candles: pd.DataFrame, symbol: str, as_of: str) -> bytes:
    cutoff = pd.Timestamp(as_of)
    source = candles.loc[candles["symbol"].eq(symbol)].sort_values("open_time_utc")
    completed = source.loc[source["close_time_utc"] <= cutoff].tail(250)
    current = source.loc[source["open_time_utc"] == cutoff.floor("h")].iloc[[0]]
    rows = pd.concat([completed, current], ignore_index=True)
    payload = []
    for row in rows.itertuples(index=False):
        open_ms = int(pd.Timestamp(row.open_time_utc).value // 1_000_000)
        close_ms = int((pd.Timestamp(row.close_time_utc) - pd.Timedelta(milliseconds=1)).value // 1_000_000)
        payload.append([open_ms, row.open, row.high, row.low, row.close, row.volume, close_ms, 0, 0, 0, 0, 0])
    return json.dumps(payload).encode("utf-8")


def _invoke(args, *, repository_root: Path, clock: str, transport=None):
    return run.main(
        args,
        repository_root=repository_root,
        clock=lambda: pd.Timestamp(clock),
        transport=transport,
    )


@pytest.fixture(scope="module")
def prepared_run(tmp_path_factory):
    repository_root = tmp_path_factory.mktemp("predictor-cli")
    config_path = repository_root / "experiments" / "predictor_v0" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    candles = _candles()
    assert _invoke(
        ["fetch", "--config", "experiments/predictor_v0/config.json"],
        repository_root=repository_root,
        clock="2025-01-07T00:01:00Z",
        transport=_archive_transport(candles),
    ) == 0
    assert _invoke(
        ["develop", "--config", "experiments/predictor_v0/config.json", "--run-id", "dev-001"],
        repository_root=repository_root,
        clock="2025-01-08T00:00:00Z",
    ) == 0
    offline_path = repository_root / "offline-candles.csv.gz"
    candles.to_csv(offline_path, index=False, compression="gzip")
    return {"root": repository_root, "config": config_path, "run_dir": repository_root / "artifacts" / "predictor_v0" / "dev-001", "candles": offline_path}


def _result(capsys):
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.err


def test_fetch_and_offline_develop_write_reproducible_development_only_artifacts(prepared_run, capsys):
    run_dir = prepared_run["run_dir"]
    expected = {
        "model.json", "model-metadata.json", "feature-schema.json", "development-predictions.csv.gz",
        "development-opportunities.csv.gz", "metrics.json", "report.md", "run-manifest.json", "config.json", "agent-log.md",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["reload_verification"]["passed"] is True
    assert manifest["hashes"]["model_metadata"]
    assert all(not Path(value).is_absolute() for value in manifest["paths"].values())
    data_manifest = json.loads((prepared_run["root"] / "data" / "predictor_v0" / "data-manifest.json").read_text(encoding="utf-8"))
    assert all(not Path(record["path"]).is_absolute() for record in data_manifest["archives"])
    assert "holdout" not in json.dumps(metrics).lower()
    report = (run_dir / "report.md").read_text(encoding="utf-8").lower()
    assert "development-only" in report
    assert "train" in report and "development" in report
    assert "holdout" in report and "no holdout metrics" in report
    assert "tree:" in report and "ridge:" in report and "zero:" in report


def test_all_nonabsolute_cli_paths_resolve_against_repository_root(prepared_run, capsys):
    root = prepared_run["root"]
    exit_code = _invoke(
        [
            "forecast",
            "--config", "experiments/predictor_v0/config.json",
            "--run-dir", "artifacts/predictor_v0/dev-001",
            "--symbol", "BTCUSDT",
            "--candles", "offline-candles.csv.gz",
            "--as-of", "2025-01-08T05:01:00Z",
        ],
        repository_root=root,
        clock="2025-01-08T05:01:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code == 0
    assert payload["status"] == "PAPER_ONLY"
    assert payload["replay"] is True
    assert payload["feature_cutoff_utc"] == "2025-01-08T05:00:00Z"
    assert payload["model_sha256"]
    assert payload["feature_schema_sha256"]
    assert payload["experiment_id"] == "crypto_4h_tree_v0"
    assert payload["schema_version"] == 1
    assert payload["forecast_net_return_assumption"]
    assert "forecast_net_return" not in payload
    assert "confidence" not in payload


def test_invalid_config_still_creates_error_manifest_for_valid_run_id(tmp_path, capsys):
    config_path = tmp_path / "bad-config.json"
    config_path.write_text("{not-json", encoding="utf-8")
    exit_code = _invoke(
        ["develop", "--config", "bad-config.json", "--run-id", "invalid-config"],
        repository_root=tmp_path,
        clock="2025-01-08T00:00:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload["code"] == "CONFIG_INVALID"
    manifest_path = tmp_path / "artifacts" / "predictor_v0" / "invalid-config" / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ERROR"
    assert manifest["code"] == "CONFIG_INVALID"
    assert str(tmp_path) not in manifest["error"]


def test_invalid_run_id_is_a_structured_nonzero_cli_error(tmp_path, capsys):
    exit_code = _invoke(
        ["develop", "--config", "missing.json", "--run-id", "bad/id"],
        repository_root=tmp_path,
        clock="2025-01-08T00:00:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload == {"status": "ERROR", "code": "RUN_ID_INVALID", "action": "NO_TRADE"}


def test_forecast_rejects_unsupported_symbol_with_parseable_nonzero_error(prepared_run, capsys):
    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "DOGEUSDT", "--candles", str(prepared_run["candles"]), "--as-of", "2025-01-08T04:01:00Z"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T04:01:00Z",
    )
    payload, stderr = _result(capsys)
    assert exit_code != 0
    assert payload == {"status": "ERROR", "code": "UNSUPPORTED_SYMBOL", "action": "NO_TRADE"}
    assert "unsupported" in stderr.lower()


@pytest.mark.parametrize(
    ("as_of", "expected_code"),
    [("2025-01-08T04:01:00Z", "DATA_PENDING"), ("2025-01-08T04:02:00Z", "DATA_PENDING"), ("2025-01-08T04:03:00Z", "STALE_DATA")],
)
def test_forecast_distinguishes_missing_latest_boundary_from_stale_data(prepared_run, tmp_path, capsys, as_of, expected_code):
    candles = pd.read_csv(prepared_run["candles"])
    missing = candles.loc[~((candles["symbol"] == "BTCUSDT") & (candles["close_time_utc"] == "2025-01-08 04:00:00+00:00"))]
    path = tmp_path / "missing-latest.csv.gz"
    missing.to_csv(path, index=False, compression="gzip")
    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "BTCUSDT", "--candles", str(path), "--as-of", as_of],
        repository_root=prepared_run["root"],
        clock=as_of,
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload["code"] == expected_code
    assert payload["action"] == "NO_TRADE"


def test_forecast_rejects_malformed_metadata(prepared_run, tmp_path, capsys):
    run_dir = tmp_path / "bad-run"
    shutil.copytree(prepared_run["run_dir"], run_dir)
    (run_dir / "model-metadata.json").write_text("{not-json", encoding="utf-8")
    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(run_dir), "--symbol", "BTCUSDT", "--candles", str(prepared_run["candles"]), "--as-of", "2025-01-08T04:01:00Z"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T04:01:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload["code"] == "MODEL_SCHEMA_ERROR"


@pytest.mark.parametrize("hash_name", ["candles", "config"])
def test_develop_refuses_fitting_when_fetched_data_integrity_changes(prepared_run, capsys, hash_name):
    manifest_path = prepared_run["root"] / "data" / "predictor_v0" / "data-manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["hashes"][hash_name] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        exit_code = _invoke(
            ["develop", "--config", "experiments/predictor_v0/config.json", "--run-id", f"tampered-{hash_name}"],
            repository_root=prepared_run["root"],
            clock="2025-01-08T00:00:00Z",
        )
    finally:
        manifest_path.write_text(original, encoding="utf-8")
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload["code"] == "DATA_INTEGRITY_ERROR"
    run_manifest = json.loads((prepared_run["root"] / "artifacts" / "predictor_v0" / f"tampered-{hash_name}" / "run-manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["status"] == "ERROR"


@pytest.mark.parametrize("hash_name", ["model", "feature_schema"])
def test_forecast_refuses_run_when_recorded_artifact_hash_changes(prepared_run, tmp_path, capsys, hash_name):
    run_dir = tmp_path / f"bad-{hash_name}"
    shutil.copytree(prepared_run["run_dir"], run_dir)
    manifest_path = run_dir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hashes"][hash_name] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(run_dir), "--symbol", "BTCUSDT", "--candles", str(prepared_run["candles"]), "--as-of", "2025-01-08T04:01:00Z"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T04:01:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload["code"] == "MODEL_INTEGRITY_ERROR"


def test_forecast_refuses_validly_shaped_edited_metadata(prepared_run, tmp_path, capsys):
    run_dir = tmp_path / "edited-metadata"
    shutil.copytree(prepared_run["run_dir"], run_dir)
    metadata_path = run_dir / "model-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_cutoff_utc"] = "2024-01-01T00:00:00Z"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(run_dir), "--symbol", "BTCUSDT", "--candles", str(prepared_run["candles"]), "--as-of", "2025-01-08T04:01:00Z"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T04:01:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload["code"] == "MODEL_INTEGRITY_ERROR"


def test_live_forecast_requests_251_and_retains_250_completed_rows(prepared_run, capsys, monkeypatch):
    as_of = "2025-01-14T04:01:00Z"
    candles = _candles(periods=320)
    calls = []

    def transport(url):
        calls.append(url)
        return _api_payload(candles, "BTCUSDT", as_of)

    captured_lengths = []
    original_build_features = run.build_features

    def capture_build_features(frame, **kwargs):
        captured_lengths.append(len(frame))
        return original_build_features(frame, **kwargs)

    monkeypatch.setattr(run, "build_features", capture_build_features)

    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "BTCUSDT"],
        repository_root=prepared_run["root"],
        clock=as_of,
        transport=transport,
    )
    payload, _ = _result(capsys)
    assert exit_code == 0
    assert "limit=251" in calls[0]
    assert captured_lengths == [250]
    assert payload["status"] == "PAPER_ONLY"
    assert payload["replay"] is False


def test_live_source_failure_is_structured_as_source_unavailable(prepared_run, capsys):
    def unavailable(_url):
        raise OSError("network unavailable")

    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "BTCUSDT"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T04:01:00Z",
        transport=unavailable,
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload == {"status": "ERROR", "code": "SOURCE_UNAVAILABLE", "action": "NO_TRADE"}


def test_live_api_error_payload_is_structured_as_source_unavailable(prepared_run, capsys):
    def api_error(_url):
        return b'{"code": -1003, "msg": "too many requests"}'

    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "BTCUSDT"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T04:01:00Z",
        transport=api_error,
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload["code"] == "SOURCE_UNAVAILABLE"


def test_malformed_offline_candles_are_structured_as_invalid_candles(prepared_run, tmp_path, capsys):
    path = tmp_path / "malformed.csv"
    path.write_text("symbol,close\nBTCUSDT,100\n", encoding="utf-8")
    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "BTCUSDT", "--candles", str(path), "--as-of", "2025-01-08T04:01:00Z"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T04:01:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code != 0
    assert payload["code"] == "INVALID_CANDLES"


def test_forecast_expired_opportunity_never_emits_long(prepared_run, capsys):
    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "BTCUSDT", "--candles", str(prepared_run["candles"]), "--as-of", "2025-01-08T04:06:00Z"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T04:06:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code == 0
    assert payload["action"] == "NO_TRADE"
    assert payload["reason"] == "SIGNAL_EXPIRED"


def test_forecast_off_schedule_is_never_long_and_exposes_old_model_age(prepared_run, capsys):
    exit_code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "BTCUSDT", "--candles", str(prepared_run["candles"]), "--as-of", "2025-01-08T05:01:00Z"],
        repository_root=prepared_run["root"],
        clock="2025-01-08T05:01:00Z",
    )
    payload, _ = _result(capsys)
    assert exit_code == 0
    assert payload["status"] == "PAPER_ONLY"
    assert payload["replay"] is True
    assert payload["action"] == "NO_TRADE"
    assert payload["reason"] == "OUTSIDE_OPPORTUNITY_WINDOW"
    assert payload["model_age_days"] == pytest.approx((pd.Timestamp("2025-01-08T05:01:00Z") - pd.Timestamp("2025-01-05T12:00:00Z")).total_seconds() / 86_400)
    assert "model_age_seconds" not in payload
    assert "confidence" not in payload


@pytest.mark.parametrize(("key", "value"), [
    ("horizon_hours", 8), ("max_threads", 8), ("seed", 7),
    ("fee_bps_per_side", 0.0), ("slippage_bps_per_side", 0.0),
    ("minimum_predicted_net_return", 0.0), ("interval", "4h"),
    ("venue", "another_venue"), ("mode", "live"), ("initial_equity", 20000.0),
    ("opportunity_hours_utc", [0, 8, 16]), ("horizon_hours", True),
])
def test_forecast_rejects_mutated_fixed_contract(prepared_run, tmp_path, capsys, key, value):
    config = _config()
    config[key] = value
    path = tmp_path / "mutated-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    code = _invoke(
        ["forecast", "--config", str(path), "--run-dir", str(prepared_run["run_dir"]),
         "--symbol", "BTCUSDT", "--candles", str(prepared_run["candles"]), "--as-of", "2025-01-08T04:01:00Z"],
        repository_root=prepared_run["root"], clock="2025-01-08T04:01:00Z",
    )
    payload, _ = _result(capsys)
    assert code == 2
    assert payload["code"] == "CONFIG_INVALID"
    assert "reference_exit_utc" not in payload


@pytest.mark.parametrize("changed", ["config", "code"])
def test_forecast_binds_current_inputs_to_completed_run(prepared_run, tmp_path, capsys, monkeypatch, changed):
    path = prepared_run["config"]
    if changed == "config":
        config = _config()
        config["train_end_utc"] = "2025-01-05T00:00:00Z"
        path = tmp_path / "changed-dates.json"
        path.write_text(json.dumps(config), encoding="utf-8")
    else:
        monkeypatch.setattr(run, "_code_hash", lambda: "0" * 64)
    code = _invoke(
        ["forecast", "--config", str(path), "--run-dir", str(prepared_run["run_dir"]),
         "--symbol", "BTCUSDT", "--candles", str(prepared_run["candles"]), "--as-of", "2025-01-08T04:01:00Z"],
        repository_root=prepared_run["root"], clock="2025-01-08T04:01:00Z",
    )
    payload, _ = _result(capsys)
    assert code == 2
    assert payload["code"] == "MODEL_INTEGRITY_ERROR"


def test_as_of_without_offline_candles_is_rejected_before_network(prepared_run, capsys):
    calls = []
    code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]),
         "--symbol", "BTCUSDT", "--as-of", "2025-01-08T04:01:00Z"],
        repository_root=prepared_run["root"], clock="2026-09-07T00:00:00Z", transport=lambda url: calls.append(url),
    )
    payload, _ = _result(capsys)
    assert code == 2
    assert payload["code"] == "REPLAY_REQUIRES_CANDLES"
    assert calls == []


def test_current_time_live_signal_still_expires(prepared_run, capsys):
    now = "2025-01-14T04:06:00Z"
    code = _invoke(
        ["forecast", "--config", str(prepared_run["config"]), "--run-dir", str(prepared_run["run_dir"]), "--symbol", "BTCUSDT"],
        repository_root=prepared_run["root"], clock=now,
        transport=lambda url: _api_payload(_candles(320), "BTCUSDT", now),
    )
    payload, _ = _result(capsys)
    assert code == 0
    assert payload["replay"] is False
    assert payload["action"] == "NO_TRADE"
    assert payload["reason"] == "SIGNAL_EXPIRED"


def test_development_forecast_population_includes_partial_terminal_delay_rows(prepared_run):
    directory = prepared_run["run_dir"]
    predictions = pd.read_csv(directory / "development-predictions.csv.gz")
    partial = predictions["delayed_entry_utc"].notna() & predictions["delayed_exit_utc"].isna()
    assert partial.sum() == 2
    metrics = json.loads((directory / "metrics.json").read_text())["development"]
    manifest = json.loads((directory / "run-manifest.json").read_text())
    for arm in ("tree", "ridge", "zero"):
        assert metrics["forecast_metrics"][arm]["pooled"]["count"] == len(predictions) == manifest["sample_rows"]["development"]
    assert metrics["screening"]["label"] == "PIPELINE_ONLY"
    assert metrics["screening"]["criteria"]
    assert "PIPELINE_ONLY" in (directory / "report.md").read_text()
    assert "buy_and_hold" in metrics["policy_metrics"]


def test_cli_caps_native_libraries_before_import_even_with_larger_inherited_limits():
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    probe = "from experiments.predictor_v0 import run; import json; from threadpoolctl import threadpool_info; print(json.dumps(threadpool_info()))"
    result = subprocess.run([sys.executable, "-c", probe], cwd=run.REPOSITORY_ROOT,
                            env={**os.environ, **dict.fromkeys(names, "32")}, capture_output=True, text=True, check=True)
    pools = json.loads(result.stdout)
    assert pools
    assert all(0 < pool["num_threads"] <= 4 for pool in pools)


@pytest.mark.parametrize(("period", "source_type", "bad_time"), [
    ("2025-01", "monthly", "2024-12-31T23:00:00Z"),
    ("2025-01-01", "daily", "2025-01-02T00:00:00Z"),
    ("2025-01", "monthly", "2025-01-01T00:30:00Z"),
])
def test_fetch_validates_archive_period_and_grid_before_clipping(tmp_path, monkeypatch, capsys, period, source_type, bad_time):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()))
    candles = _candles()
    extra = candles.iloc[[0]].copy()
    extra["open_time_utc"] = pd.Timestamp(bad_time)
    extra["close_time_utc"] = pd.Timestamp(bad_time) + pd.Timedelta(hours=1)
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(_archive_payload(pd.concat([candles, extra]), "BTCUSDT"))
    monkeypatch.setattr(run, "download_archives", lambda *args, **kwargs: [{
        "path": str(archive), "url": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/fixture.zip",
        "unit": "us", "period": period, "source_type": source_type,
    }])
    code = _invoke(["fetch", "--config", str(config_path)], repository_root=tmp_path, clock="2025-01-08T00:00:00Z")
    payload, stderr = _result(capsys)
    assert code == 2
    assert "outside" in stderr or "hour grid" in stderr
    assert not (tmp_path / "data/predictor_v0/data-manifest.json").exists()


def test_fetch_aggregates_duplicates_before_final_deduplication(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()))
    candles = _candles()
    source = pd.concat([candles, candles.iloc[[0, 1]]], ignore_index=True)
    assert _invoke(["fetch", "--config", str(path)], repository_root=tmp_path,
                   clock="2025-01-08T00:00:00Z", transport=_archive_transport(source)) == 0
    payload, _ = _result(capsys)
    assert payload["quality"]["duplicate_rows_dropped"] == 2
    assert payload["quality"]["row_count"] == 288


@pytest.mark.parametrize("failure", ["coverage", "transport", "publication"])
def test_fetch_failure_keeps_successful_bundle_and_persists_diagnostics(prepared_run, tmp_path, capsys, monkeypatch, failure):
    root = tmp_path
    data_dir = root / "data/predictor_v0"
    data_dir.mkdir(parents=True)
    for name in ("data-manifest.json", "candles.csv.gz"):
        shutil.copyfile(prepared_run["root"] / "data/predictor_v0" / name, data_dir / name)
    before = {name: (data_dir / name).read_bytes() for name in ("data-manifest.json", "candles.csv.gz")}
    path = root / "config.json"
    path.write_text(json.dumps(_config()))
    def unavailable(url):
        raise OSError("offline fixture source unavailable")
    if failure == "publication":
        original_replace = Path.replace
        def interrupted_replace(source, target):
            if source.name == "data-manifest.json" and source.parent != data_dir:
                raise OSError("simulated manifest publication failure")
            return original_replace(source, target)
        monkeypatch.setattr(Path, "replace", interrupted_replace)
    transport = unavailable if failure == "transport" else _archive_transport(_candles().iloc[10:] if failure == "coverage" else _candles())
    assert _invoke(["fetch", "--config", str(path)], repository_root=root,
                   clock="2025-01-08T00:00:00Z", transport=transport) == 2
    payload, _ = _result(capsys)
    assert {name: (data_dir / name).read_bytes() for name in before} == before
    diagnostics = json.loads((root / payload["diagnostics_path"]).read_text())
    assert diagnostics["status"] == "ERROR"
    if failure == "coverage":
        assert diagnostics["code"] == "RAW_COVERAGE_INSUFFICIENT"
        assert diagnostics["quality"]["coverage"]["BTCUSDT"]["missing_count"] == 10
        assert diagnostics["quality"]["missing_timestamps"]["BTCUSDT"]


@pytest.mark.parametrize(("full", "counts", "delay", "mse", "label"), [
    (False, [50, 50], 0.01, 0.1, "PIPELINE_ONLY"),
    (True, [49, 50], 0.01, 0.1, "INSUFFICIENT_EVIDENCE"),
    (True, [24, 76], 0.01, 0.1, "INSUFFICIENT_EVIDENCE"),
    (True, [25, 75], 0.01, 0.1, "CANDIDATE_FOR_FORWARD_TEST"),
    (True, [25, 75], 0.0, 0.1, "NO_EDGE"),
    (True, [25, 75], 0.01, 0.2, "NO_EDGE"),
])
def test_screening_uses_fixed_counts_and_strict_positive_edge(full, counts, delay, mse, label):
    assets = {symbol: {"selected_trades": count, "average_net_return": 0.01} for symbol, count in zip(run.SYMBOL_MAPPING, counts)}
    comparison = {
        "forecast_metrics": {name: {"pooled": {"mse": value}} for name, value in (("tree", mse), ("zero", 0.2), ("ridge", 0.3))},
        "strategy_summaries": {"tree": {
            "selected_opportunities": sum(counts), "per_asset_metrics": assets,
            "doubled_costs_fixed_selections": {"per_asset_metrics": assets},
            "delayed_reference_price_proxy": {"average_net_return": delay},
        }},
    }
    result = run._screen_development(comparison, full)
    assert result["label"] == label
    assert result["criteria"]["pooled_selected_count"]["required"] == ">= 100"


def test_daily_fallback_fetch_parses_each_declared_day_and_clips_only_after_validation(tmp_path, capsys):
    from urllib.error import HTTPError
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()))
    candles = _candles(744)
    payloads = {}
    for (symbol, day), frame in candles.groupby(["symbol", candles["open_time_utc"].dt.strftime("%Y-%m-%d")]):
        payloads[f"{symbol}-1h-{day}"] = _archive_payload(frame, symbol)
    def transport(url):
        if "/monthly/" in url:
            raise HTTPError(url, 404, "absent", None, io.BytesIO())
        body = payloads[url.rsplit("/", 1)[-1].split(".zip")[0]]
        return hashlib.sha256(body).hexdigest().encode() if url.endswith(".CHECKSUM") else body
    assert _invoke(["fetch", "--config", str(path)], repository_root=tmp_path,
                   clock="2025-02-01T00:00:00Z", transport=transport) == 0
    payload, _ = _result(capsys)
    assert payload["quality"]["row_count"] == 288
    manifest = json.loads((tmp_path / "data/predictor_v0/data-manifest.json").read_text())
    assert len(manifest["archives"]) == 62
    assert all(record["source_type"] == "daily" for record in manifest["archives"])
