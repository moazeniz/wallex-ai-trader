"""Offline tests for the Wallex WebSocket stream (fake frames in the formats observed live on 2026-10-22)."""
import asyncio
import json
import time
from datetime import datetime

import pytest

from experiments.wallex_arena import feed as feed_mod
from experiments.wallex_arena.feed import MarketFeed
from experiments.wallex_arena.ws_feed import (SYMBOLS_PER_CONN, WallexStream, channels_for, parse_depth,
                                              parse_trades, split_channel)


def depth_frame(sym, side, levels):
    kind = "buyDepth" if side == "bids" else "sellDepth"
    msg = [{"price": p, "quantity": q, "sum": p * q} for p, q in levels]
    return json.dumps({"push": {"channel": f"orderbook:{sym}@{kind}", "pub": {"data": {"message": msg}}}})


def trade_frame(sym, price, qty, buy, stamp):
    return json.dumps({"push": {"channel": f"public:{sym}@trade", "pub": {"data": {
        "isBuyOrder": buy, "price": str(price), "quantity": str(qty), "timestamp": stamp}}}})


def ts_of(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def rest_book(sym, bid, ask, ts):
    mid = (bid + ask) / 2
    return {"symbol": sym, "bid": bid, "ask": ask, "mid": mid, "spread": (ask - bid) / mid,
            "asks": [(ask, 1.0), (ask + 1, 2.0)], "bids": [(bid, 1.0), (bid - 1, 2.0)], "ts": ts}


@pytest.fixture
def feed():
    f = MarketFeed()
    assert f.stream is None
    f.stream = WallexStream(f)  # attached but never started: no network
    yield f
    f.pool.shutdown(wait=False)


def test_parse_real_formats():
    raw = {"message": [{"price": 22126695860, "quantity": 0.00189775, "sum": 41990937.07},
                       {"price": 22126695861, "quantity": 0.00451942, "sum": 99999831.8},
                       {"price": 22115821568, "quantity": 0, "sum": 0}]}
    assert parse_depth(raw, bids=True) == [(22126695861.0, 0.00451942), (22126695860.0, 0.00189775)]
    asks = parse_depth({"message": [{"price": 3, "quantity": 1}, {"price": 2, "quantity": 1}]}, bids=False)
    assert asks == [(2.0, 1.0), (3.0, 1.0)]
    many = {"message": [{"price": 100 + i, "quantity": 1} for i in range(50)]}
    assert len(parse_depth(many, bids=False)) == 40
    rows = parse_trades({"isBuyOrder": True, "price": "358001", "quantity": "276.4", "timestamp": "2026-10-22T12:15:28Z"})
    assert rows == [(ts_of("2026-10-22T12:15:28Z"), 358001.0, 276.4, True)]
    assert split_channel("orderbook:BTCTMN@buyDepth") == ("BTCTMN", "buyDepth")
    assert split_channel("public:USDTTMN@trade") == ("USDTTMN", "trade")
    assert channels_for("BTCTMN") == ["orderbook:BTCTMN@buyDepth", "orderbook:BTCTMN@sellDepth", "public:BTCTMN@trade"]


def test_batched_frame_and_ping(feed):
    feed.books["BTCTMN"] = rest_book("BTCTMN", 100.0, 110.0, time.time() - 5)
    frame = "\n".join([depth_frame("BTCTMN", "bids", [(101, 1), (100, 2)]), "{}",
                       trade_frame("BTCTMN", 105, 0.5, False, "2026-10-22T12:14:45Z")])
    pings = feed.stream.on_frame(frame)
    assert pings == 1
    assert feed.books["BTCTMN"]["bid"] == 101.0
    assert list(feed.trades["BTCTMN"]) == [(ts_of("2026-10-22T12:14:45Z"), 105.0, 0.5, False)]
    assert feed.stream.counts["BTCTMN"] == 2 and feed.stream.parse_errors == 0


def test_per_side_merge(feed):
    t0 = time.time() - 10
    feed.books["ETHTMN"] = rest_book("ETHTMN", 100.0, 110.0, t0)
    old_asks = feed.books["ETHTMN"]["asks"]
    now = time.time()
    feed.stream.on_frame(depth_frame("ETHTMN", "bids", [(102, 1), (103, 3), (101, 2)]), recv_ts=now)
    b = feed.books["ETHTMN"]
    assert b["bids"] == [(103.0, 3.0), (102.0, 1.0), (101.0, 2.0)]   # best first
    assert b["asks"] == old_asks and b["ask"] == 110.0                   # other side untouched
    assert b["bids_ts"] == now and b["asks_ts"] == t0 and b["ts"] == t0  # book is as old as its stalest side
    assert b["mid"] == (103 + 110) / 2 and b["src"] == "ws"
    later = now + 1
    feed.stream.on_frame(depth_frame("ETHTMN", "asks", [(108, 1), (109, 1)]), recv_ts=later)
    b = feed.books["ETHTMN"]
    assert b["bids"][0] == (103.0, 3.0) and b["asks"] == [(108.0, 1.0), (109.0, 1.0)]
    assert b["ts"] == now and b["asks_ts"] == later


def test_crossing_and_unknown_symbol(feed):
    feed.books["XRPTMN"] = rest_book("XRPTMN", 100.0, 110.0, time.time() - 5)  # asks 110, 111
    feed.stream.on_frame(depth_frame("XRPTMN", "bids", [(110.5, 1)]))
    assert feed.books["XRPTMN"]["asks"] == [(111.0, 2.0)] and feed.books["XRPTMN"]["bid"] == 110.5
    before = feed.books["XRPTMN"]
    feed.stream.on_frame(depth_frame("XRPTMN", "bids", [(200, 1)]))  # would empty the asks: not published
    assert feed.books["XRPTMN"] is before
    feed.stream.on_frame(depth_frame("NEWTMN", "bids", [(1, 1)]))
    assert "NEWTMN" not in feed.books  # the first book always comes from REST


def _no_network(monkeypatch, feed, depth_fn):
    monkeypatch.setattr(feed_mod, "http_json", lambda *a, **k: [])
    feed.refresh_meta = lambda force=False: None
    feed.focus = ["SOLTMN"]
    feed._depth = depth_fn


def test_rest_does_not_overwrite_newer_stream_book(feed, monkeypatch):
    feed.books["SOLTMN"] = rest_book("SOLTMN", 100.0, 110.0, time.time() - 20)

    def depth(sym):  # the stream delivers both sides while the REST request is in flight
        feed.stream.on_frame(depth_frame(sym, "bids", [(105, 1)]))
        feed.stream.on_frame(depth_frame(sym, "asks", [(106, 1)]))
        return sym, [(110.0, 1.0)], [(100.0, 1.0)], time.time()

    _no_network(monkeypatch, feed, depth)
    feed.poll(with_trades=False)
    assert (feed.books["SOLTMN"]["bid"], feed.books["SOLTMN"]["ask"]) == (105.0, 106.0)
    assert feed.rest_skipped == 1


def test_rest_overwrites_older_or_half_stale_stream_book(feed, monkeypatch):
    feed.books["SOLTMN"] = rest_book("SOLTMN", 100.0, 110.0, time.time() - 20)
    feed.stream.on_frame(depth_frame("SOLTMN", "bids", [(105, 1)]))  # only the bid side is fresh
    assert feed.books["SOLTMN"]["src"] == "ws"
    _no_network(monkeypatch, feed, lambda sym: (sym, [(107.0, 1.0)], [(104.0, 1.0)], time.time()))
    feed.poll(with_trades=False)
    b = feed.books["SOLTMN"]
    assert (b["bid"], b["ask"]) == (104.0, 107.0) and "src" not in b and feed.rest_skipped == 0
    # both sides from the stream but before the REST request started: REST still wins
    feed.stream.on_frame(depth_frame("SOLTMN", "bids", [(105, 1)]), recv_ts=time.time() - 1)
    feed.stream.on_frame(depth_frame("SOLTMN", "asks", [(106, 1)]), recv_ts=time.time() - 1)
    feed.poll(with_trades=False)
    assert feed.books["SOLTMN"]["ask"] == 107.0


def test_rest_only_mode_unchanged(monkeypatch):
    f = MarketFeed(websocket=False)
    monkeypatch.setattr(feed_mod, "http_json", lambda *a, **k: [])
    f.refresh_meta = lambda force=False: None
    f.focus = ["SOLTMN"]
    f._depth = lambda sym: (sym, [(107.0, 1.0)], [(104.0, 1.0)], 123.0)
    f._trades = lambda sym: (sym, [(2.0, 1.0, 1.0, True), (1.0, 1.0, 1.0, False)])
    f.poll()
    assert f.books["SOLTMN"] == {"symbol": "SOLTMN", "bid": 104.0, "ask": 107.0, "mid": 105.5, "spread": 3 / 105.5,
                                 "asks": [(107.0, 1.0)], "bids": [(104.0, 1.0)], "ts": 123.0}
    assert list(f.trades["SOLTMN"]) == [(1.0, 1.0, 1.0, False), (2.0, 1.0, 1.0, True)]
    f.pool.shutdown(wait=False)


def test_trade_dedup_with_rest_and_order(feed, monkeypatch):
    s1, s2, s0 = "2026-10-22T12:15:28Z", "2026-10-22T12:15:30Z", "2026-10-22T12:15:20Z"
    feed.stream.on_frame(trade_frame("XRPTMN", 358001, 276.4, True, s1))
    feed.stream.on_frame(trade_frame("XRPTMN", 358001, 276.4, True, s1))  # duplicate push
    feed.stream.on_frame(trade_frame("XRPTMN", 359048, 24.5, False, s2))
    assert len(feed.trades["XRPTMN"]) == 2
    # REST returns the same trades in its own format plus an older one the stream missed (e.g. during a reconnect)
    rest_rows = [{"symbol": "XRPTMN", "quantity": "276.4000000000000000", "price": "358001.0000000000000000",
                  "isBuyOrder": True, "timestamp": s1},
                 {"symbol": "XRPTMN", "quantity": "24.5000000000000000", "price": "359048.0000000000000000",
                  "isBuyOrder": False, "timestamp": s2},
                 {"symbol": "XRPTMN", "quantity": "1.0000000000000000", "price": "357000.0000000000000000",
                  "isBuyOrder": False, "timestamp": s0}]
    monkeypatch.setattr(feed_mod, "http_json", lambda *a, **k: {"result": {"latestTrades": rest_rows}})
    sym, rows = feed._trades("XRPTMN")
    feed._add_trades(sym, rows, keep_sorted=True)
    got = list(feed.trades["XRPTMN"])
    assert [t[0] for t in got] == [ts_of(s0), ts_of(s1), ts_of(s2)]  # ascending, no duplicates
    assert len(feed._trade_keys["XRPTMN"]) == 3
    assert feed.trades_since("XRPTMN", ts_of(s0)) == got[1:]


def test_resubscribe_on_focus_change():
    s = WallexStream(feed=None)
    syms = [f"S{i:02d}TMN" for i in range(25)]
    s.set_symbols(syms)
    added, removed = s._assign()
    assert len(added) == 25 and not removed
    assert [len(c.symbols) for c in s.conns] == [SYMBOLS_PER_CONN, 25 - SYMBOLS_PER_CONN]
    assert all(3 * len(c.symbols) <= 64 for c in s.conns)
    assert s.set_symbols(syms) is False  # unchanged set: no work

    sent = {}

    class FakeWS:
        def __init__(self, idx):
            self.idx = idx

        async def send(self, text):
            sent.setdefault(self.idx, []).append(json.loads(text))

        async def close(self):
            sent.setdefault(self.idx, []).append("closed")

    class FakeTask:
        def cancel(self):
            pass

    for c in s.conns:
        c.ws, c.connected, c.task = FakeWS(c.idx), True, FakeTask()
    new = syms[3:24] + ["NEWATMN", "NEWBTMN"]  # drop S00-S02 and S24, add two
    assert s.set_symbols(new)
    asyncio.run(s._reconcile())
    ops0 = [(op, m[op]["channel"]) for m in sent[0] for op in ("subscribe", "unsubscribe") if op in m]
    unsub = {ch for op, ch in ops0 if op == "unsubscribe"}
    sub = {ch for op, ch in ops0 if op == "subscribe"}
    assert unsub == {ch for x in ("S00TMN", "S01TMN", "S02TMN") for ch in channels_for(x)}
    assert sub == {ch for x in ("NEWATMN", "NEWBTMN") for ch in channels_for(x)}
    assert {x for c in s.conns for x in c.symbols} == set(new)
    assert all(3 * len(c.symbols) <= 64 for c in s.conns)
    assert any(m.get("unsubscribe", {}).get("channel") == "public:S24TMN@trade" for m in sent[1] if m != "closed")

    # replies: success marks the channel subscribed, error 106 is recorded and not counted
    c = s.conns[0]
    ids = {v[1]: k for k, v in c.pending.items() if v[0] == "subscribe"}
    s.on_frame(json.dumps({"id": ids["public:NEWATMN@trade"], "subscribe": {}}), c)
    s.on_frame(json.dumps({"id": ids["public:NEWBTMN@trade"], "error": {"code": 106, "message": "limit exceeded"}}), c)
    assert c.subscribed == {"public:NEWATMN@trade"}
    assert s.errors[-1][2] == 106 and s.health()["channels"] == 1

    s.set_symbols(["NEWATMN"])  # shrinking empties connection 1: it is closed and dropped
    asyncio.run(s._reconcile())
    assert len(s.conns) == 1 and "closed" in sent[1]


def test_health_shape(feed):
    feed.books["BTCTMN"] = rest_book("BTCTMN", 100.0, 110.0, time.time())
    feed.stream.set_symbols(["BTCTMN", "ETHTMN"])
    feed.stream.on_frame(depth_frame("BTCTMN", "asks", [(109, 1)]))
    h = feed.stream.health()
    assert h["symbols"] == 2 and h["silent_symbols"] == ["ETHTMN"] and h["book_updates"] == 1
    assert h["reconnects"] == 0 and h["symbol_age_max"] is not None
