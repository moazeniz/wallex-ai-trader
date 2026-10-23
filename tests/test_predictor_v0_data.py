import hashlib
import io
import json
import zipfile
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import pytest

from experiments.predictor_v0 import run
from experiments.predictor_v0.data import (
    FEATURE_COLUMNS,
    _quality_report,
    build_features,
    build_samples,
    download_archives,
    drop_incomplete_candles,
    load_candles,
    normalize_klines,
    parse_timestamp,
    validate_candles,
)


UTC = "2025-01-01T00:00:00Z"


def raw_kline(open_time=1735689600000, close_time=1735693199999):
    return [open_time, "100", "110", "90", "105", "7", close_time, "0", 0, "0", "0", "0"]


def candle_frame(times=(UTC,)):
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "open_time_utc": pd.to_datetime(list(times), utc=True),
            "close_time_utc": pd.to_datetime(list(times), utc=True) + pd.Timedelta(hours=1),
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 7.0,
        }
    )


@pytest.fixture
def hourly_market_candles():
    """130 unique, monotonic hourly bars per supported symbol."""
    times = pd.date_range("2025-01-01T00:00:00Z", periods=130, freq="h")
    frames = []
    for symbol, base in (("BTCUSDT", 40.0), ("ETHUSDT", 400.0)):
        opens = base + np.arange(len(times), dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "open_time_utc": times,
                    "close_time_utc": times + pd.Timedelta(hours=1),
                    "open": opens,
                    "high": opens + 2.0,
                    "low": opens - 1.0,
                    "close": opens + 1.0,
                    "volume": 100.0 + np.arange(len(times), dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_parse_timestamp_explicit_ms_and_us_are_equivalent():
    ms = parse_timestamp(pd.Series([1735689600000]), "ms")
    us = parse_timestamp(pd.Series([1735689600000000]), "us")
    assert ms.iloc[0] == pd.Timestamp(UTC)
    assert us.iloc[0] == ms.iloc[0]
    with pytest.raises(ValueError, match="ms or us"):
        parse_timestamp(pd.Series([1735689600000]), "ns")


def test_normalize_klines_enforces_raw_hour_boundary_and_column_contract():
    normalized = normalize_klines(pd.DataFrame([raw_kline()]), "BTCUSDT", "ms")
    assert list(normalized.columns) == [
        "symbol", "open_time_utc", "close_time_utc", "open", "high", "low", "close", "volume",
    ]
    assert normalized.loc[0, "close_time_utc"] == pd.Timestamp("2025-01-01T01:00:00Z")
    invalid = raw_kline(close_time=1735693200000)
    with pytest.raises(ValueError, match="close time"):
        normalize_klines(pd.DataFrame([invalid]), "BTCUSDT", "ms")


def test_normalize_klines_records_in_interval_raw_close_anomaly_without_correction():
    frame = pd.DataFrame(
        [raw_kline(open_time=1679659200000, close_time=1679661581646)]
    )
    normalized = normalize_klines(frame, "BTCUSDT", "ms")
    assert normalized.loc[0, "close_time_utc"] == pd.Timestamp("2023-03-24T13:00:00Z")
    assert normalized.attrs["raw_close_time_anomalies"] == [
        {
            "symbol": "BTCUSDT",
            "row_index": 0,
            "open_time_utc": "2023-03-24T12:00:00Z",
            "observed_close_time_utc": "2023-03-24T12:39:41.646000Z",
            "expected_close_time_utc": "2023-03-24T12:59:59.999000Z",
            "delta_ms": -1218353,
        }
    ]
    json.dumps(normalized.attrs["raw_close_time_anomalies"])


def test_normalize_klines_rejects_raw_close_outside_open_hour_interval():
    outside = raw_kline(open_time=1679659200000, close_time=1679662800000)
    with pytest.raises(ValueError, match="interval"):
        normalize_klines(pd.DataFrame([outside]), "BTCUSDT", "ms")


def test_normalize_klines_rejects_raw_close_before_open_time():
    before_open = raw_kline(open_time=1679659200000, close_time=1679659199999)
    with pytest.raises(ValueError, match="interval"):
        normalize_klines(pd.DataFrame([before_open]), "BTCUSDT", "ms")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.assign(high=99.0), "high"),
        (lambda frame: frame.assign(volume=-1.0), "volume"),
        (lambda frame: frame.assign(close=np.inf), "finite"),
        (lambda frame: frame.assign(symbol="DOGEUSDT"), "symbol"),
        (lambda frame: frame.assign(open_time_utc=pd.Timestamp("2024-12-31T23:00:00Z")), "outside"),
    ],
)
def test_validate_candles_rejects_invalid_rows(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_candles(
            mutator(candle_frame()),
            pd.Timestamp(UTC),
            pd.Timestamp("2025-01-01T02:00:00Z"),
        )


def test_validate_candles_removes_identical_duplicates_and_rejects_conflicts():
    frame = pd.concat([candle_frame(), candle_frame()], ignore_index=True)
    report = validate_candles(frame, pd.Timestamp(UTC), pd.Timestamp("2025-01-01T01:00:00Z"))
    assert report["duplicate_rows_dropped"] == 1
    assert report["row_count"] == 1
    conflicting = frame.copy()
    conflicting.loc[1, "close"] = 106.0
    with pytest.raises(ValueError, match="conflicting duplicate"):
        validate_candles(conflicting, pd.Timestamp(UTC), pd.Timestamp("2025-01-01T01:00:00Z"))


def test_validate_candles_reports_every_gap_without_interpolation():
    frame = candle_frame((UTC, "2025-01-01T03:00:00Z"))
    report = validate_candles(frame, pd.Timestamp(UTC), pd.Timestamp("2025-01-01T04:00:00Z"))
    assert report["missing_timestamps"] == {"BTCUSDT": ["2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"]}
    assert report["missing_count"] == 2
    assert len(frame) == 2


def test_drop_incomplete_candles_keeps_only_closed_bars():
    frame = candle_frame((UTC, "2025-01-01T01:00:00Z"))
    kept = drop_incomplete_candles(frame, "2025-01-01T01:30:00Z")
    assert kept["open_time_utc"].tolist() == [pd.Timestamp(UTC)]


def zip_bytes(rows):
    payload = "\n".join(",".join(map(str, row)) for row in rows).encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1h-2024-12.csv", payload)
    return stream.getvalue()


def archive_config():
    return {
        "symbols": ["BTCUSDT"],
        "raw_start_utc": "2024-12-01T00:00:00Z",
        "development_end_utc": "2025-01-01T00:00:00Z",
        "holdout_end_utc": "2025-02-01T00:00:00Z",
    }


def http_error(url, status):
    return HTTPError(url, status, f"HTTP {status}", None, io.BytesIO())


def test_download_archives_verifies_checksum_and_reuses_valid_cache(tmp_path):
    body = zip_bytes([raw_kline(open_time=1733011200000, close_time=1733014799999)])
    digest = hashlib.sha256(body).hexdigest()
    calls = []

    def transport(url):
        calls.append(url)
        return f"{digest} BTCUSDT-1h-2024-12.zip".encode() if url.endswith(".CHECKSUM") else body

    first = download_archives(archive_config(), tmp_path, transport=transport)
    second = download_archives(archive_config(), tmp_path, transport=transport)
    assert first[0]["unit"] == "ms"
    assert first[0]["source_type"] == "monthly"
    assert first[0]["actual_sha256"] == digest
    assert first[0]["url"].endswith("BTCUSDT-1h-2024-12.zip")
    assert second[0]["cached"] is True
    assert calls.count(first[0]["url"]) == 1


def test_download_archives_rejects_checksum_mismatch(tmp_path):
    def transport(url):
        return b"0" * 64 if url.endswith(".CHECKSUM") else zip_bytes([raw_kline()])

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_archives(archive_config(), tmp_path, transport=transport)


def test_fetch_writes_raw_close_anomalies_to_data_manifest(tmp_path):
    anomaly = raw_kline(open_time=1679659200000, close_time=1679661581646)
    body = zip_bytes([anomaly])
    digest = hashlib.sha256(body).hexdigest()
    config_path = tmp_path / "config.json"
    config = {
        "symbols": ["BTCUSDT"],
        "raw_start_utc": "2023-03-24T12:00:00Z",
        "development_end_utc": "2023-03-24T13:00:00Z",
        "holdout_end_utc": "2023-03-24T14:00:00Z",
        "_path": str(config_path),
    }
    config_path.write_text(json.dumps({key: value for key, value in config.items() if key != "_path"}), encoding="utf-8")

    def transport(url):
        return f"{digest} BTCUSDT-1h-2023-03.zip".encode() if url.endswith(".CHECKSUM") else body

    result = run._fetch(config, tmp_path, False, transport)
    manifest = json.loads((tmp_path / "data" / "predictor_v0" / "data-manifest.json").read_text(encoding="utf-8"))
    expected = {
        "symbol": "BTCUSDT",
        "archive_url": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2023-03.zip",
        "archive_period": "2023-03",
        "archive_source_type": "monthly",
        "row_index": 0,
        "open_time_utc": "2023-03-24T12:00:00Z",
        "observed_close_time_utc": "2023-03-24T12:39:41.646000Z",
        "expected_close_time_utc": "2023-03-24T12:59:59.999000Z",
        "delta_ms": -1218353,
    }
    assert result["quality"]["raw_close_time_anomalies"] == [expected]
    assert manifest["quality"]["raw_close_time_anomalies"] == [expected]
    json.dumps(manifest["quality"]["raw_close_time_anomalies"])


def test_download_archives_falls_back_only_after_monthly_zip_404(tmp_path):
    body = zip_bytes([raw_kline(open_time=1733011200000, close_time=1733014799999)])
    digest = hashlib.sha256(body).hexdigest()
    monthly_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-12.zip"

    def transport(url):
        if url == f"{monthly_url}.CHECKSUM":
            return f"{digest} BTCUSDT-1h-2024-12.zip".encode()
        if url == monthly_url:
            raise http_error(url, 404)
        if url.endswith(".CHECKSUM"):
            return f"{digest} daily.zip".encode()
        return body

    records = download_archives(archive_config(), tmp_path, transport=transport)
    assert len(records) == 31
    assert {record["source_type"] for record in records} == {"daily"}


def test_download_archives_rejects_monthly_checksum_404_without_daily_fallback(tmp_path):
    calls = []

    def transport(url):
        calls.append(url)
        if url.endswith(".CHECKSUM"):
            raise http_error(url, 404)
        return zip_bytes([raw_kline()])

    with pytest.raises(RuntimeError, match=r"HTTP 404: .*monthly.*\.CHECKSUM"):
        download_archives(archive_config(), tmp_path, transport=transport)
    assert len(calls) == 2
    assert all("/daily/" not in url for url in calls)
    assert not list(tmp_path.glob("*.zip"))


@pytest.mark.parametrize("failure", [URLError("temporary"), TimeoutError("temporary"), 503])
def test_download_archives_retries_transient_failures_at_most_three_times(tmp_path, failure):
    body = zip_bytes([raw_kline(open_time=1733011200000, close_time=1733014799999)])
    digest = hashlib.sha256(body).hexdigest()
    checksum_calls = 0

    def transport(url):
        nonlocal checksum_calls
        if url.endswith(".CHECKSUM"):
            checksum_calls += 1
            if checksum_calls <= 3:
                raise http_error(url, failure) if isinstance(failure, int) else failure
            return f"{digest} BTCUSDT-1h-2024-12.zip".encode()
        return body

    with pytest.raises(RuntimeError, match="fetch failed"):
        download_archives(archive_config(), tmp_path, transport=transport)
    assert checksum_calls == 3


def test_download_archives_falls_back_when_monthly_zip_and_checksum_are_both_absent(tmp_path):
    calls = []
    body = zip_bytes([raw_kline()])
    digest = hashlib.sha256(body).hexdigest()

    def transport(url):
        calls.append(url)
        if "/monthly/" in url:
            raise http_error(url, 404)
        return digest.encode() if url.endswith(".CHECKSUM") else body

    records = download_archives(archive_config(), tmp_path, transport=transport)
    assert len(records) == 31
    assert {record["source_type"] for record in records} == {"daily"}
    assert len([url for url in calls if "/monthly/" in url]) == 2
    assert all(record["expected_sha256"] == record["actual_sha256"] for record in records)


def test_off_hour_grid_is_rejected_and_coverage_counts_only_matched_expected_hours():
    frame = candle_frame((UTC, "2025-01-01T01:30:00Z"))
    with pytest.raises(ValueError, match="hour grid"):
        validate_candles(frame, pd.Timestamp(UTC), pd.Timestamp("2025-01-01T02:00:00Z"))
    quality = _quality_report(frame, pd.Timestamp(UTC), pd.Timestamp("2025-01-01T02:00:00Z"), 0)
    assert quality["coverage"]["BTCUSDT"]["observed_count"] == 1
    assert quality["coverage"]["BTCUSDT"]["coverage"] == 0.5


@pytest.mark.parametrize("status", [403, 451])
def test_download_archives_stops_immediately_on_access_denial(tmp_path, status):
    calls = []

    def transport(url):
        calls.append(url)
        raise http_error(url, status)

    with pytest.raises(RuntimeError, match=rf"HTTP {status}"):
        download_archives(archive_config(), tmp_path, transport=transport)
    assert len(calls) == 1


def test_load_candles_reads_gzip_csv_and_applies_utc_schema_checks(tmp_path):
    path = tmp_path / "candles.csv.gz"
    candle_frame().to_csv(path, index=False, compression="gzip")
    loaded = load_candles(path, as_of_utc="2025-01-01T02:00:00Z")
    assert loaded["open_time_utc"].dtype == "datetime64[us, UTC]"
    assert loaded["symbol"].tolist() == ["BTCUSDT"]


def test_build_samples_uses_next_open_and_exact_four_hour_exit(hourly_market_candles):
    candles = hourly_market_candles.copy()
    btc_rows = candles.index[candles["symbol"] == "BTCUSDT"]
    entry_row, exit_row = btc_rows[61], btc_rows[65]
    candles.loc[entry_row, ["open", "high", "low", "close"]] = [100.0, 102.0, 99.0, 101.0]
    candles.loc[exit_row:, ["open", "high", "low", "close"]] += 6.0
    candles.loc[exit_row, ["open", "high", "low", "close"]] = [110.0, 112.0, 109.0, 111.0]

    sample = build_samples(candles).query("symbol == 'BTCUSDT'").iloc[11]

    assert sample["reference_entry_price"] == 100.0
    assert sample["reference_exit_price"] == 110.0
    assert sample["target_return"] == pytest.approx(0.10)
    assert sample["reference_entry_utc"] == sample["feature_cutoff_utc"]
    assert sample["reference_exit_utc"] == sample["feature_cutoff_utc"] + pd.Timedelta(hours=4)
    assert sample["label_end_utc"] == sample["feature_cutoff_utc"] + pd.Timedelta(hours=4)
    assert sample["delayed_entry_utc"] == sample["feature_cutoff_utc"] + pd.Timedelta(hours=1)
    assert sample["delayed_exit_utc"] == sample["feature_cutoff_utc"] + pd.Timedelta(hours=5)
    assert sample["delayed_entry_price"] == 102.0
    assert sample["delayed_exit_price"] == 112.0
    assert sample["target_valid"] is True


def test_build_features_do_not_change_when_later_prices_change(hourly_market_candles):
    cutoff = pd.Timestamp("2025-01-03T23:00:00Z")
    original = build_features(hourly_market_candles)
    changed = hourly_market_candles.copy()
    future = changed["open_time_utc"] >= cutoff
    changed.loc[future, ["open", "high", "low", "close"]] *= 10.0
    updated = build_features(changed)

    pd.testing.assert_frame_equal(
        original.loc[(slice(None), slice(None, cutoff)), :],
        updated.loc[(slice(None), slice(None, cutoff)), :],
    )


def test_as_of_cutoff_excludes_in_progress_and_future_close_candles(hourly_market_candles):
    as_of_utc = pd.Timestamp("2025-01-06T08:30:00Z")

    features = build_features(hourly_market_candles, as_of_utc=as_of_utc)
    samples = build_samples(hourly_market_candles, as_of_utc=as_of_utc)

    assert features.index.get_level_values("feature_cutoff_utc").max() == pd.Timestamp("2025-01-06T08:00:00Z")
    assert samples["feature_cutoff_utc"].max() == pd.Timestamp("2025-01-06T08:00:00Z")


def test_shifting_eth_prices_leaves_btc_features_and_labels_unchanged(hourly_market_candles):
    original = build_samples(hourly_market_candles).query("symbol == 'BTCUSDT'").reset_index(drop=True)
    shifted = hourly_market_candles.copy()
    eth = shifted["symbol"] == "ETHUSDT"
    shifted.loc[eth, ["open", "high", "low", "close"]] += 1000.0
    updated = build_samples(shifted).query("symbol == 'BTCUSDT'").reset_index(drop=True)

    pd.testing.assert_frame_equal(original, updated)


def test_gap_resets_feature_and_label_windows(hourly_market_candles):
    missing_open = pd.Timestamp("2025-01-04T03:00:00Z")
    gapped = hourly_market_candles.loc[
        ~((hourly_market_candles["symbol"] == "BTCUSDT") & (hourly_market_candles["open_time_utc"] == missing_open))
    ]
    features = build_features(gapped)
    samples = build_samples(gapped)
    btc = samples.query("symbol == 'BTCUSDT'")
    pre_gap = btc.loc[btc["feature_cutoff_utc"] == missing_open]
    post_gap_warmup = features.loc[("BTCUSDT", slice(missing_open + pd.Timedelta(hours=1), missing_open + pd.Timedelta(hours=49))), :]

    assert pre_gap.iloc[0]["target_valid"] is False
    assert pd.isna(pre_gap.iloc[0]["delayed_entry_utc"])
    assert pd.isna(pre_gap.iloc[0]["delayed_exit_utc"])
    assert pd.isna(pre_gap.iloc[0]["delayed_entry_price"])
    assert pd.isna(pre_gap.iloc[0]["delayed_exit_price"])
    assert btc["segment_id"].nunique() == 2
    assert post_gap_warmup.empty


def test_last_unlabeled_feature_row_is_retained_for_inference(hourly_market_candles):
    samples = build_samples(hourly_market_candles, horizon_hours=4)
    btc = samples.query("symbol == 'BTCUSDT'")
    final = btc.iloc[-1]

    assert not btc.tail(5)["target_valid"].any()
    assert final["target_valid"] is False
    assert pd.isna(final["target_return"])
    assert final["feature_cutoff_utc"] == pd.Timestamp("2025-01-06T10:00:00Z")


def test_build_features_are_ordered_when_input_columns_are_permuted(hourly_market_candles):
    ordered = build_features(hourly_market_candles)
    permuted = build_features(hourly_market_candles.loc[:, list(reversed(hourly_market_candles.columns))])

    assert list(ordered.columns) == FEATURE_COLUMNS
    assert ordered.index.names == ["symbol", "feature_cutoff_utc"]
    assert ordered.index.is_unique
    assert ordered.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(ordered, permuted)
