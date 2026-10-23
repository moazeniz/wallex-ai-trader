"""Causal, auditable public-candle ingestion primitives for predictor v0."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd


SUPPORTED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})
NORMALIZED_COLUMNS = [
    "symbol",
    "open_time_utc",
    "close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
]
BINANCE_MONTHLY_ROOT = "https://data.binance.vision/data/spot/monthly/klines"
BINANCE_DAILY_ROOT = "https://data.binance.vision/data/spot/daily/klines"
_NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "return_1h",
    "return_4h",
    "return_12h",
    "return_24h",
    "volatility_12h",
    "volatility_24h",
    "price_vs_sma20",
    "price_vs_sma50",
    "range_fraction",
    "volume_ratio24",
    "hour_sin",
    "hour_cos",
    "is_eth",
]


def parse_timestamp(values: pd.Series, unit: str) -> pd.Series:
    """Parse epoch values only when their millisecond/microsecond unit is explicit."""
    if unit not in {"ms", "us"}:
        raise ValueError("timestamp unit must be explicitly ms or us")
    numeric = pd.to_numeric(values, errors="raise")
    if pd.isna(numeric).any():
        raise ValueError("timestamps must not be missing")
    return pd.to_datetime(numeric, unit=unit, utc=True)


def normalize_klines(frame: pd.DataFrame, symbol: str, unit: str) -> pd.DataFrame:
    """Convert Binance positional or named rows to the public hourly candle schema."""
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"unsupported symbol: {symbol}")
    if unit not in {"ms", "us"}:
        raise ValueError("timestamp unit must be explicitly ms or us")

    named = {str(column).lower(): column for column in frame.columns}
    required = {"open_time", "close_time", "open", "high", "low", "close", "volume"}
    if required.issubset(named):
        raw = pd.DataFrame(
            {
                "open_time": frame[named["open_time"]],
                "close_time": frame[named["close_time"]],
                "open": frame[named["open"]],
                "high": frame[named["high"]],
                "low": frame[named["low"]],
                "close": frame[named["close"]],
                "volume": frame[named["volume"]],
            }
        )
    elif frame.shape[1] == 12:
        raw = frame.iloc[:, [0, 6, 1, 2, 3, 4, 5]].copy()
        raw.columns = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
    else:
        raise ValueError("klines must be Binance 12-column rows or a clearly named equivalent")

    open_time = parse_timestamp(raw["open_time"], unit)
    raw_close_time = parse_timestamp(raw["close_time"], unit)
    interval = pd.Timedelta(hours=1)
    precision = pd.Timedelta(milliseconds=1 if unit == "ms" else 0, microseconds=1 if unit == "us" else 0)
    expected_raw_close = open_time + interval - precision
    interval_end = open_time + interval
    outside_interval = (raw_close_time < open_time) | (raw_close_time >= interval_end)
    if outside_interval.any():
        raise ValueError("Binance raw close time must be inside the one-hour interval [open_time, open_time + 1h)")
    anomalies = []
    for position in range(len(raw)):
        observed = raw_close_time.iloc[position]
        expected = expected_raw_close.iloc[position]
        if observed != expected:
            delta_ms = (observed - expected) / pd.Timedelta(milliseconds=1)
            delta = int(delta_ms) if float(delta_ms).is_integer() else float(delta_ms)
            anomalies.append(
                {
                    "symbol": symbol,
                    "row_index": int(position),
                    "open_time_utc": _iso_utc(open_time.iloc[position]),
                    "observed_close_time_utc": _iso_utc(observed),
                    "expected_close_time_utc": _iso_utc(expected),
                    "delta_ms": delta,
                }
            )

    normalized = pd.DataFrame(
        {
            "symbol": symbol,
            "open_time_utc": open_time,
            "close_time_utc": open_time + interval,
            **{column: pd.to_numeric(raw[column], errors="raise") for column in _NUMERIC_COLUMNS},
        }
    )
    _validate_values(normalized)
    normalized = normalized.sort_values(["symbol", "open_time_utc"], kind="stable").reset_index(drop=True)
    normalized.attrs["raw_close_time_anomalies"] = anomalies
    return normalized


def validate_candles(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Validate a public candle frame and return JSON-safe coverage and gap diagnostics."""
    _, report = _clean_and_validate(frame, start, end)
    return report


def drop_incomplete_candles(frame: pd.DataFrame, as_of_utc: pd.Timestamp | str) -> pd.DataFrame:
    """Keep only candles whose exclusive close boundary was known at ``as_of_utc``."""
    if "close_time_utc" not in frame:
        raise ValueError("missing required column: close_time_utc")
    cutoff = _utc_timestamp(as_of_utc, "as_of_utc")
    result = frame.copy()
    result["close_time_utc"] = pd.to_datetime(result["close_time_utc"], utc=True, errors="raise")
    return result.loc[result["close_time_utc"] <= cutoff].reset_index(drop=True)


def build_features(candles: pd.DataFrame, *, as_of_utc: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Build causal features from candles closed by the optional UTC as-of cutoff."""
    prepared = _prepare_segmented_candles(candles, as_of_utc=as_of_utc)
    return _build_features_from_prepared(prepared)


def build_samples(
    candles: pd.DataFrame,
    horizon_hours: int = 4,
    *,
    as_of_utc: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Attach label metadata to features from candles closed by an optional UTC cutoff."""
    if not isinstance(horizon_hours, int) or isinstance(horizon_hours, bool) or horizon_hours < 1:
        raise ValueError("horizon_hours must be a positive integer")

    prepared = _prepare_segmented_candles(candles, as_of_utc=as_of_utc)
    features = _build_features_from_prepared(prepared)
    metadata = prepared[["symbol", "close_time_utc", "segment_id"]].copy()
    metadata = metadata.rename(columns={"close_time_utc": "feature_cutoff_utc"})
    metadata["label_end_utc"] = metadata["feature_cutoff_utc"] + pd.Timedelta(hours=horizon_hours)

    for _, group in prepared.groupby(["symbol", "segment_id"], sort=False):
        index = group.index
        entry_price = group["open"].shift(-1)
        exit_price = group["open"].shift(-(horizon_hours + 1))
        metadata.loc[index, "reference_entry_utc"] = group["open_time_utc"].shift(-1)
        metadata.loc[index, "reference_exit_utc"] = group["open_time_utc"].shift(-(horizon_hours + 1))
        metadata.loc[index, "reference_entry_price"] = entry_price
        metadata.loc[index, "reference_exit_price"] = exit_price
        metadata.loc[index, "delayed_entry_utc"] = group["open_time_utc"].shift(-2)
        metadata.loc[index, "delayed_exit_utc"] = group["open_time_utc"].shift(-(horizon_hours + 2))
        metadata.loc[index, "delayed_entry_price"] = group["open"].shift(-2)
        metadata.loc[index, "delayed_exit_price"] = group["open"].shift(-(horizon_hours + 2))

    valid = (
        metadata[["reference_entry_price", "reference_exit_price"]].notna().all(axis=1)
        & np.isfinite(metadata[["reference_entry_price", "reference_exit_price"]]).all(axis=1)
        & (metadata[["reference_entry_price", "reference_exit_price"]] > 0).all(axis=1)
    )
    metadata["target_valid"] = valid.astype(object)
    metadata["target_return"] = metadata["reference_exit_price"] / metadata["reference_entry_price"] - 1.0
    metadata.loc[~valid, "target_return"] = np.nan

    metadata = metadata.set_index(["symbol", "feature_cutoff_utc"])
    return features.join(metadata, how="left").reset_index()


def load_candles(path: Path, as_of_utc: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Read CSV/gzip CSV, normalize its schema, and reject invalid public candles."""
    frame = pd.read_csv(path)
    cleaned, _ = _clean_and_validate(frame, None, None)
    if as_of_utc is not None:
        cleaned = drop_incomplete_candles(cleaned, as_of_utc)
    return cleaned


def download_archives(
    config: dict,
    raw_dir: Path,
    include_holdout: bool = False,
    transport: Callable[[str], bytes] | None = None,
) -> list[dict]:
    """Download verified Binance monthly archives, falling back to daily archives on monthly 404."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    symbols = config.get("symbols", [])
    if not symbols or any(symbol not in SUPPORTED_SYMBOLS for symbol in symbols):
        raise ValueError("config symbols must be supported public symbols")
    start = _utc_timestamp(config["raw_start_utc"], "raw_start_utc")
    end_key = "holdout_end_utc" if include_holdout else "development_end_utc"
    end = _utc_timestamp(config[end_key], end_key)
    if start >= end:
        raise ValueError("archive period start must precede end")

    records: list[dict] = []
    first_month = start.normalize().replace(day=1)
    last_month = end.normalize().replace(day=1)
    for symbol in symbols:
        for month in pd.date_range(start=first_month, end=last_month, freq="MS", tz="UTC"):
            if month >= end:
                continue
            unit = "us" if month >= pd.Timestamp("2025-01-01T00:00:00Z") else "ms"
            month_label = month.strftime("%Y-%m")
            url = f"{BINANCE_MONTHLY_ROOT}/{symbol}/1h/{symbol}-1h-{month_label}.zip"
            try:
                records.append(_fetch_verified(url, raw_dir / f"{symbol}-1h-{month_label}.zip", unit, month_label, "monthly", transport))
            except _NotFound as error:
                if error.url != url:
                    raise RuntimeError(f"archive fetch failed with HTTP {error.status}: {error.url}") from error
                records.extend(_download_daily_month(symbol, month, raw_dir, unit, transport))
    return records


class _NotFound(Exception):
    def __init__(self, url: str, status: int):
        self.url = url
        self.status = status
        super().__init__(f"HTTP {status}: {url}")


def _download_daily_month(symbol: str, month: pd.Timestamp, raw_dir: Path, unit: str, transport) -> list[dict]:
    records = []
    for day in pd.date_range(month, month + pd.offsets.MonthBegin(1), inclusive="left", freq="D", tz="UTC"):
        label = day.strftime("%Y-%m-%d")
        url = f"{BINANCE_DAILY_ROOT}/{symbol}/1h/{symbol}-1h-{label}.zip"
        records.append(_fetch_verified(url, raw_dir / f"{symbol}-1h-{label}.zip", unit, label, "daily", transport))
    return records


def _fetch_verified(url: str, path: Path, unit: str, period: str, source_type: str, transport) -> dict:
    checksum_url = f"{url}.CHECKSUM"
    try:
        expected_hash = _checksum_hash(_fetch(checksum_url, transport))
    except _NotFound as error:
        # Distinguish an absent archive (daily fallback) from an unverifiable one.
        _fetch(url, transport)
        raise RuntimeError(f"existing archive has no checksum; HTTP 404: {checksum_url}") from error
    cached = path.exists() and _sha256_path(path) == expected_hash
    if not cached:
        payload = _fetch(url, transport)
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"checksum mismatch for {url}")
        path.write_bytes(payload)
    actual_hash = _sha256_path(path)
    if actual_hash != expected_hash:
        raise ValueError(f"checksum mismatch for cached {path}")
    return {
        "url": url,
        "checksum_url": checksum_url,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "unit": unit,
        "period": period,
        "path": str(path),
        "source_type": source_type,
        "cached": cached,
    }


def _fetch(url: str, transport) -> bytes:
    for attempt in range(3):
        try:
            if transport is not None:
                payload = transport(url)
            else:
                with urlopen(url, timeout=30) as response:  # nosec B310 - fixed Binance archive host
                    payload = response.read()
            if not isinstance(payload, bytes):
                raise TypeError("transport must return bytes")
            return payload
        except HTTPError as error:
            if error.code == 404:
                raise _NotFound(url, error.code) from error
            if error.code in {403, 451}:
                raise RuntimeError(f"archive access stopped with HTTP {error.code}: {url}") from error
            if error.code < 500 or attempt == 2:
                raise RuntimeError(f"archive fetch failed with HTTP {error.code}: {url}") from error
        except (URLError, TimeoutError, ConnectionError) as error:
            if attempt == 2:
                raise RuntimeError(f"archive fetch failed after retries: {url}") from error
        if transport is None:
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"archive fetch failed after retries: {url}")


def _clean_and_validate(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> tuple[pd.DataFrame, dict]:
    missing = [column for column in NORMALIZED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    cleaned = frame.loc[:, NORMALIZED_COLUMNS].copy()
    cleaned["symbol"] = cleaned["symbol"].astype(str)
    for column in ("open_time_utc", "close_time_utc"):
        cleaned[column] = pd.to_datetime(cleaned[column], utc=True, errors="raise")
    for column in _NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    _validate_values(cleaned)
    start_ts = _utc_timestamp(start, "start") if start is not None else None
    end_ts = _utc_timestamp(end, "end") if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts >= end_ts:
        raise ValueError("start must precede end")
    if start_ts is not None and (cleaned["open_time_utc"] < start_ts).any():
        raise ValueError("candle timestamp falls outside requested interval")
    if end_ts is not None and (cleaned["open_time_utc"] >= end_ts).any():
        raise ValueError("candle timestamp falls outside requested interval")
    if not cleaned["open_time_utc"].eq(cleaned["open_time_utc"].dt.floor("h")).all():
        raise ValueError("candle timestamps must lie on the UTC hour grid")
    if not (cleaned["close_time_utc"] == cleaned["open_time_utc"] + pd.Timedelta(hours=1)).all():
        raise ValueError("close_time_utc must equal open_time_utc plus one hour")

    keys = ["symbol", "open_time_utc"]
    duplicate_rows_dropped = 0
    duplicate_mask = cleaned.duplicated(keys, keep=False)
    if duplicate_mask.any():
        for _, group in cleaned.loc[duplicate_mask].groupby(keys, sort=False):
            if len(group.drop_duplicates()) != 1:
                raise ValueError("conflicting duplicate candle rows")
        duplicate_rows_dropped = int(cleaned.duplicated(keep="first").sum())
        cleaned = cleaned.drop_duplicates(keep="first")
    cleaned = cleaned.sort_values(keys, kind="stable").reset_index(drop=True)
    return cleaned, _quality_report(cleaned, start_ts, end_ts, duplicate_rows_dropped)


def _prepare_segmented_candles(
    candles: pd.DataFrame,
    *,
    as_of_utc: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Validate candles, apply the optional completion cutoff, and segment them."""
    prepared, _ = _clean_and_validate(candles, None, None)
    if as_of_utc is not None:
        prepared = drop_incomplete_candles(prepared, as_of_utc)
    expected_interval = pd.Timedelta(hours=1)
    prepared["segment_id"] = prepared.groupby("symbol", sort=False)["open_time_utc"].transform(
        lambda times: times.diff().ne(expected_interval).cumsum() - 1
    )
    return prepared


def _build_features_from_prepared(prepared: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (symbol, _), group in prepared.groupby(["symbol", "segment_id"], sort=False):
        close = group["close"]
        returns = {f"return_{hours}h": close / close.shift(hours) - 1.0 for hours in (1, 4, 12, 24)}
        volume_mean = group["volume"].rolling(24, min_periods=24).mean().replace(0.0, np.nan)
        hours = group["close_time_utc"].dt.hour
        features = pd.DataFrame(
            {
                **returns,
                "volatility_12h": returns["return_1h"].rolling(12, min_periods=12).std(ddof=0),
                "volatility_24h": returns["return_1h"].rolling(24, min_periods=24).std(ddof=0),
                "price_vs_sma20": close / close.rolling(20, min_periods=20).mean() - 1.0,
                "price_vs_sma50": close / close.rolling(50, min_periods=50).mean() - 1.0,
                "range_fraction": (group["high"] - group["low"]) / close,
                "volume_ratio24": group["volume"] / volume_mean,
                "hour_sin": np.sin(2.0 * np.pi * hours / 24.0),
                "hour_cos": np.cos(2.0 * np.pi * hours / 24.0),
                "is_eth": 1.0 if symbol == "ETHUSDT" else 0.0,
            },
            index=group.index,
        )
        features = features.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()
        features.index = pd.MultiIndex.from_arrays(
            [group.loc[features.index, "symbol"], group.loc[features.index, "close_time_utc"]],
            names=["symbol", "feature_cutoff_utc"],
        )
        frames.append(features)

    if not frames:
        index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "feature_cutoff_utc"])
        return pd.DataFrame(columns=FEATURE_COLUMNS, index=index, dtype=float)
    return pd.concat(frames).sort_index()


def _validate_values(frame: pd.DataFrame) -> None:
    symbols = set(frame["symbol"])
    unsupported = symbols - SUPPORTED_SYMBOLS
    if unsupported:
        raise ValueError(f"unsupported symbol: {sorted(unsupported)[0]}")
    values = frame[_NUMERIC_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OHLCV values must be finite")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume must be nonnegative")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("high must be at least open and close")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any() or (frame["low"] > frame["high"]).any():
        raise ValueError("low must not exceed open, close, or high")


def _quality_report(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None, duplicate_rows_dropped: int) -> dict:
    missing_timestamps: dict[str, list[str]] = {}
    coverage: dict[str, dict] = {}
    for symbol, group in frame.groupby("symbol", sort=True):
        observed = group["open_time_utc"]
        range_start = start if start is not None else observed.min()
        range_end = end if end is not None else observed.max() + pd.Timedelta(hours=1)
        expected = pd.date_range(range_start, range_end, freq="h", inclusive="left", tz="UTC")
        gaps = expected.difference(observed)
        serialized_gaps = [stamp.isoformat().replace("+00:00", "Z") for stamp in gaps]
        missing_timestamps[symbol] = serialized_gaps
        expected_count = len(expected)
        matched_count = len(expected.intersection(observed))
        coverage[symbol] = {
            "expected_count": expected_count,
            "observed_count": matched_count,
            "missing_count": len(serialized_gaps),
            "coverage": float(matched_count / expected_count) if expected_count else 1.0,
        }
    return {
        "row_count": int(len(frame)),
        "symbol_count": int(frame["symbol"].nunique()),
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "missing_count": int(sum(len(gaps) for gaps in missing_timestamps.values())),
        "missing_timestamps": missing_timestamps,
        "coverage": coverage,
    }


def _utc_timestamp(value: pd.Timestamp | str, name: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{name} is required")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def _iso_utc(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _checksum_hash(payload: bytes) -> str:
    match = re.search(rb"\b([0-9a-fA-F]{64})\b", payload)
    if match is None:
        raise ValueError("invalid Binance checksum payload")
    return match.group(1).decode().lower()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
