"""Read-only, source-labelled Wallex context with an optional TradingView cross-check.

All indicators consume complete candles available at the snapshot cutoff. No order,
wallet, private TradingView account, or LLM methods exist in this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import subprocess
import time

from experiments.wallex_challenge.api import PublicAPI
from experiments.wallex_challenge.core import Candle, Market, validate_candles
from .risk import valid_tmn_symbol

NODE = Path("/opt/wallex/runtime/node/bin/node")
BRIDGE = Path(__file__).resolve().parent / "tradingview" / "fetch.mjs"
TV_PACKAGE_VERSION = "4.0.0"


@dataclass(frozen=True)
class MarketConfig:
    history_days: int = 8
    minimum_hourly_bars: int = 72
    maximum_bar_age_seconds: int = 5400
    tradingview_enabled: bool = True
    tradingview_count: int = 200
    tradingview_timeout_seconds: float = 25
    tradingview_cache_seconds: int = 3600
    tradingview_symbols: dict[str,str] | None = None

    def __post_init__(self):
        if self.tradingview_symbols is not None:
            if not isinstance(self.tradingview_symbols,dict):raise ValueError('TradingView symbols must be a mapping')
            for symbol,target in self.tradingview_symbols.items():
                if not valid_tmn_symbol(symbol) or not isinstance(target,str) or not re.fullmatch(r'BINANCE:[A-Z0-9]{1,20}USDT',target):
                    raise ValueError('Invalid TMN/TradingView symbol mapping')
            object.__setattr__(self,'tradingview_symbols',deepcopy(self.tradingview_symbols))
        if type(self.history_days) is not int or not 4 <= self.history_days <= 60:
            raise ValueError("history_days must be 4..60")
        if type(self.minimum_hourly_bars) is not int or not 72 <= self.minimum_hourly_bars <= self.history_days * 24:
            raise ValueError("minimum_hourly_bars must be 72..requested hours")
        if type(self.maximum_bar_age_seconds) is not int or not 0 < self.maximum_bar_age_seconds <= 7200:
            raise ValueError("maximum_bar_age_seconds must be 1..7200")
        if type(self.tradingview_enabled) is not bool:
            raise ValueError("tradingview_enabled must be boolean")
        if type(self.tradingview_count) is not int or not 72 <= self.tradingview_count <= 500:
            raise ValueError("tradingview_count must be 72..500")
        if not math.isfinite(self.tradingview_timeout_seconds) or not 1 <= self.tradingview_timeout_seconds <= 30:
            raise ValueError("tradingview_timeout_seconds must be 1..30")
        if type(self.tradingview_cache_seconds) is not int or not 0 <= self.tradingview_cache_seconds <= 3600:
            raise ValueError("tradingview_cache_seconds must be 0..3600")


def _clock(now):
    value = time.time() if now is None else float(now)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid snapshot cutoff")
    return value


def _closed_rows(raw, now, interval=3600, minimum=1):
    """Reject unsorted/gapped completed data; future bars never enter features."""
    rows = []
    dropped = 0
    for row in raw:
        item = asdict(row) if is_dataclass(row) else row
        timestamp = item["t"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or int(timestamp) != timestamp:
            raise ValueError("Invalid candle timestamp")
        if timestamp + interval > now:
            dropped += 1
            continue
        rows.append(Candle(int(timestamp), *(float(item[key]) for key in ("o", "h", "l", "c", "v"))))
    validate_candles(rows, interval)
    if len(rows) < minimum:
        raise ValueError("Insufficient complete contiguous candles")
    return rows, dropped


def _ema(values, period):
    if len(values) < period:
        return None
    result = statistics.fmean(values[:period])
    alpha = 2 / (period + 1)
    for value in values[period:]:
        result += alpha * (value - result)
    return result


def _wilder(values, period):
    if len(values) < period:
        return None
    result = statistics.fmean(values[:period])
    for value in values[period:]:
        result = (result * (period - 1) + value) / period
    return result


def indicators(rows):
    """Indicators from exactly the supplied ordered, complete candle sequence."""
    if not rows:
        raise ValueError("No candles for indicators")
    closes = [row.c for row in rows]
    changes = [b - a for a, b in zip(closes, closes[1:])]
    gain = _wilder([max(0, value) for value in changes], 14)
    loss = _wilder([max(0, -value) for value in changes], 14)
    rsi = None if gain is None else (50.0 if gain == loss == 0 else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)))
    ranges = [max(row.h - row.l, abs(row.h - rows[i - 1].c), abs(row.l - rows[i - 1].c)) for i, row in enumerate(rows) if i]
    atr = _wilder(ranges, 14)
    ema12, ema26 = _ema(closes, 12), _ema(closes, 26)
    logarithms = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    average_volume = statistics.fmean(row.v for row in rows[-25:-1]) if len(rows) >= 25 else None
    result = {
        "bars": len(rows), "last_price": closes[-1], "last_candle_open": rows[-1].t,
        "rsi14": rsi, "atr14": atr, "atr_fraction": atr / closes[-1] if atr is not None else None,
        "ema12": ema12, "ema26": ema26,
        "ema_trend": None if ema26 is None else ("up" if ema12 > ema26 else ("down" if ema12 < ema26 else "flat")),
        "realized_volatility_24_bars": statistics.pstdev(logarithms[-24:]) if len(logarithms) >= 24 else None,
        "relative_volume_24_bars": rows[-1].v / average_volume if average_volume else None,
    }
    for horizon in (1, 4, 24):
        result["return_%s_bars" % horizon] = closes[-1] / closes[-horizon - 1] - 1 if len(closes) > horizon else None
    return result


def aggregate_four_hour(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.t // 14400 * 14400, []).append(row)
    return [Candle(t, bars[0].o, max(row.h for row in bars), min(row.l for row in bars), bars[-1].c, sum(row.v for row in bars))
            for t, bars in sorted(grouped.items())
            if len(bars) == 4 and [row.t for row in bars] == [t + i * 3600 for i in range(4)]]


def technical_summary(raw, now):
    rows, _ = _closed_rows(raw, now, minimum=25)
    hourly = indicators(rows)
    hourly.update({"return_1h": hourly["return_1_bars"], "return_4h": hourly["return_4_bars"], "return_24h": hourly["return_24_bars"],
                   "realized_volatility_24h": hourly["realized_volatility_24_bars"], "relative_volume_24h": hourly["relative_volume_24_bars"]})
    four = aggregate_four_hour(rows)
    return {"interval_seconds": 3600, "last_closed_at": rows[-1].t + 3600,
            "one_hour": hourly, "four_hour": indicators(four) if four else None,
            "four_hour_policy": "UTC-aligned bins containing four complete contiguous hourly candles only",
            "indicator_policy": "Wilder RSI/ATR14; SMA-seeded EMA12/26; population volatility of unannualized log returns; no imputation"}


def order_book(payload, observed_at):
    if not payload.get("success"):
        raise ValueError("Depth response unsuccessful")
    result = payload["result"]
    sides = {}
    for raw_name, name, reverse in (("bid", "bids", True), ("ask", "asks", False)):
        levels = []
        for row in result[raw_name]:
            price, quantity = float(row["price"]), float(row["quantity"])
            if not all(math.isfinite(value) and value > 0 for value in (price, quantity)):
                raise ValueError("Order book prices and quantities must be finite and positive")
            levels.append({"price": price, "quantity": quantity})
        if not levels or len({row["price"] for row in levels}) != len(levels):
            raise ValueError("Empty or duplicate order book levels")
        sides[name] = sorted(levels, key=lambda row: row["price"], reverse=reverse)
    bid, ask = sides["bids"][0]["price"], sides["asks"][0]["price"]
    if ask <= bid:
        raise ValueError("Invalid or crossed order book")
    return {**sides, "bid": bid, "ask": ask, "spread_fraction": (ask - bid) / ((ask + bid) / 2),
            "observed_at": observed_at, "source": "wallex", "quote_currency": "TMN", "source_timestamp_available": False,
            "source_timestamp": None, "note": "REST reception time does not prove exchange quote freshness"}


def tradingview_symbol(symbol, config):
    if not valid_tmn_symbol(symbol):raise ValueError('Invalid TMN research symbol')
    target=(config.tradingview_symbols or {}).get(symbol,'BINANCE:'+symbol[:-3]+'USDT')
    if not isinstance(target,str) or not re.fullmatch(r'BINANCE:[A-Z0-9]{1,20}USDT',target):
        raise ValueError('Invalid TradingView target')
    return target


class TradingViewSource:
    """Fixed executable/script, bounded requests, no shell or account credentials."""
    def __init__(self, config=None):
        self.config = config or MarketConfig()
        self._cache = {}

    def fetch(self, symbol, now=None):
        now = _clock(now)
        target = tradingview_symbol(symbol,self.config)
        base = {"source": "tradingview", "symbol": target, "quote_currency": "USDT",
                "conversion_to_tmn_applied": False, "package": "@mathieuc/tradingview", "package_version": TV_PACKAGE_VERSION,
                "independent_unofficial_client": True}
        if not self.config.tradingview_enabled:
            return {**base, "status": "disabled", "candles": []}
        cached = self._cache.get(symbol)
        if cached and now // 3600 == cached[0] // 3600 and 0 <= now - cached[0] < self.config.tradingview_cache_seconds:
            return {**deepcopy(cached[1]), "cache_hit": True}
        try:
            query = {"symbol": target, "timeframe": "60", "count": self.config.tradingview_count,
                     "timeoutMs": int(max(1000, (self.config.tradingview_timeout_seconds - 2) * 1000)), "closedBefore": now}
            completed = subprocess.run([str(NODE), str(BRIDGE)], input=json.dumps(query), text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       timeout=self.config.tradingview_timeout_seconds, check=False,
                                       env={"PATH": str(NODE.parent), "LANG": "C.UTF-8"}, cwd=str(BRIDGE.parent))
            if completed.returncode:
                raise ValueError("bridge_exit_%s" % completed.returncode)
            if len(completed.stdout) > 2_000_000:
                raise ValueError("bridge_output_too_large")
            payload = json.loads(completed.stdout)
            if payload.get("symbol") != target or payload.get("quote_currency") != "USDT" or payload.get("package_version") != TV_PACKAGE_VERSION:
                raise ValueError("bridge_provenance_mismatch")
            raw = payload["candles"]
            if not 1 <= len(raw) <= self.config.tradingview_count:
                raise ValueError("bridge_candle_count_invalid")
            rows, dropped = _closed_rows(raw, now, minimum=25)
            age = now - (rows[-1].t + 3600)
            if age > self.config.maximum_bar_age_seconds:
                raise ValueError("tradingview_candles_stale")
            result = {**base, "status": "ok", "candles": [asdict(row) for row in rows],
                      "technical": technical_summary(rows, now), "fetched_at": payload.get("fetched_at"), "cache_hit": False,
                      "quality": {"interval_seconds": 3600, "validated_sorted_contiguous": True, "unfinished_candles_dropped": dropped,
                                  "age_seconds": age, "source_delays_possible": True},
                      "runtime": payload.get("runtime")}
            self._cache[symbol] = (now, deepcopy(result))
            return result
        except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
            # Never fabricate fallback candles or leak child stderr into the decision context.
            return {**base, "status": "unavailable", "candles": [], "attempted_at": now,
                    "error_kind": "timeout" if isinstance(error, subprocess.TimeoutExpired) else type(error).__name__,
                    "reason": "optional_cross_market_source_failed_validation_or_request"}


class MarketContext:
    def __init__(self, api=None, tradingview=None, config=None):
        self.config = config or MarketConfig()
        self.api = api or PublicAPI()
        self.tradingview = tradingview or TradingViewSource(self.config)

    def refresh_execution_context(self, context, now=None):
        """Refresh captured Wallex depth without changing closed-bar research evidence.

        The caller must separately check candle age/bar rollover before any new entry.
        This public read-only operation never invokes TradingView or history routes.
        """
        symbol=context.get('symbol')
        if not valid_tmn_symbol(symbol) or context.get('market',{}).get('symbol')!=symbol:
            raise ValueError('Execution refresh requires matching TMN market identity')
        Market(**context['market'])  # Preserve validated precision and minimum-size rules.
        metadata=context.get('market_metadata',{})
        if metadata.get('source')!='wallex' or metadata.get('quote_currency')!='TMN':
            raise ValueError('Execution refresh requires existing Wallex market metadata')
        analysis_at=float(context.get('analysis_observed_at',context['observed_at']))
        cutoff=_clock(now)
        if not math.isfinite(analysis_at) or analysis_at<=0 or analysis_at>cutoff+5:
            raise ValueError('Invalid analysis observation timestamp')
        raw_book=self.api.get('/v1/depth',{'symbol':symbol})
        received_at=_clock(None) if now is None else cutoff
        book=order_book(raw_book,received_at)
        result=deepcopy(context)
        result.update({'observed_at':received_at,'analysis_observed_at':analysis_at,'book':book})
        result['execution_refresh']={'source':'wallex','route':'/v1/depth','observed_at':received_at,
                                     'market_metadata':'reused_existing_validated_rules',
                                     'analysis_evidence_refreshed':False}
        # The original candle cutoff/hash remain intact; this adds execution-only provenance.
        provenance=result.setdefault('provenance',{})
        provenance['execution_book_refreshed_at']=received_at
        return result

    def snapshot(self, symbol, now=None):
        if not valid_tmn_symbol(symbol):
            raise ValueError("Invalid TMN research market symbol")
        cutoff = _clock(now)
        try:market, metadata = self.api.market(symbol)
        except KeyError as error:raise ValueError("TMN symbol is not listed in Wallex market metadata") from error
        if market.symbol != symbol:
            raise ValueError("Wallex market symbol mismatch")
        rows, quality = self.api.history(symbol, self.config.history_days, 3600)
        rows, dropped = _closed_rows(rows, cutoff, minimum=self.config.minimum_hourly_bars)
        age = cutoff - (rows[-1].t + 3600)
        if age > self.config.maximum_bar_age_seconds:
            raise ValueError("Wallex hourly candles are stale")
        raw_book = self.api.get("/v1/depth", {"symbol": symbol})
        # A caller-supplied cutoff supports deterministic replay; real snapshots record reception time.
        book_time = _clock(None) if now is None else cutoff
        book = order_book(raw_book, book_time)
        try:
            cross = self.tradingview.fetch(symbol, cutoff)
        except Exception as error:
            cross = {"source": "tradingview", "status": "unavailable", "symbol": tradingview_symbol(symbol,self.config), "quote_currency": "USDT",
                     "conversion_to_tmn_applied": False, "candles": [], "error_kind": type(error).__name__,
                     "reason": "optional_cross_market_source_failed"}
        candles = [asdict(row) for row in rows]
        content_hash = hashlib.sha256(json.dumps(candles, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {"schema_version": 1, "symbol": symbol, "observed_at": cutoff, "mode": "read_only_research",
                "market": asdict(market), "market_metadata": {"source": "wallex", "quote_currency": "TMN", "base_currency": symbol[:-3], "venue": metadata},
                "candles": candles,
                "candle_quality": {**quality, "source": "wallex", "quote_currency": "TMN", "interval_seconds": 3600,
                                   "snapshot_cutoff": cutoff, "unfinished_candles_dropped_at_snapshot": dropped,
                                   "age_seconds": age, "validated_sorted_contiguous": True, "selected_candles": len(rows)},
                "book": book, "technical": technical_summary(rows, cutoff), "cross_market": cross,
                "provenance": {"primary": "https://api.wallex.ir", "market_route": "/v1/markets", "history_route": "/v1/udf/history",
                               "book_route": "/v1/depth", "candles_sha256": content_hash, "ohlcv_kind": "observed_public_exchange_history",
                               "snapshot_cutoff": cutoff, "exchange_order_submission": False}}
