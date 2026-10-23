"""Offline tests for the Wallex arena: paper accounting, fills, maker-fill rules, risk caps, strategy signals,
AI action validation and the tamper-evident decision log. No network access."""
import json
import math
import time
from types import SimpleNamespace

import pytest

from experiments.wallex_arena.account import DecisionLog, PaperAccount, TAKER_FEE, MAKER_FEE
from experiments.wallex_arena.engine import Engine, RestingOrder, Risk
from experiments.wallex_arena.feed import Bars
from experiments.wallex_arena.strategies import LagReversion


def book(bid, ask, size=1.0):
    return {"bids": [(bid, size), (bid * 0.999, size)], "asks": [(ask, size), (ask * 1.001, size)], "bid": bid, "ask": ask,
            "mid": (bid + ask) / 2, "spread": (ask - bid) / ((ask + bid) / 2), "ts": time.time()}


def test_taker_round_trip_accounting(tmp_path):
    acct = PaperAccount("t", tmp_path)
    fill = acct.market_order("QQTMN", "BUY", book(99_000, 100_000, size=100), 1, budget_tmn=1_000_000)
    assert fill and fill["liquidity"] == "taker"
    assert math.isclose(fill["notional_tmn"] + fill["fee_tmn"], 1_000_000, rel_tol=1e-9)
    assert math.isclose(fill["fee_tmn"], fill["notional_tmn"] * TAKER_FEE["TMN"])
    assert math.isclose(acct.cash, 4_000_000)
    sell = acct.market_order("QQTMN", "SELL", book(101_000, 102_000, size=100), 1)
    expected = fill["qty"] * 101_000 * (1 - TAKER_FEE["TMN"]) - 1_000_000
    assert math.isclose(sell["pnl_tmn"], expected, rel_tol=1e-9)
    assert acct.positions() == {}
    assert math.isclose(acct.cash, 5_000_000 + expected, rel_tol=1e-12)


def test_buy_walks_levels_and_respects_impact_bound(tmp_path):
    acct = PaperAccount("t", tmp_path)
    thin = {"asks": [(100.0, 2000.0), (100.2, 2000.0), (110.0, 1e9)], "bids": [(99.0, 1e9)]}
    fill = acct.market_order("QQTMN", "BUY", thin, 1, budget_tmn=1_000_000, max_impact=0.004)
    assert math.isclose(fill["qty"], 4000.0)          # the 110 level is beyond the 0.4% bound
    assert fill["partial"]


def test_minimum_order_enforced(tmp_path):
    acct = PaperAccount("t", tmp_path)
    assert acct.market_order("QQTMN", "BUY", book(99, 100, size=1e9), 1, budget_tmn=99_000) is None
    assert acct.db.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 0


def test_maker_fee_and_decision_log_chain(tmp_path):
    acct = PaperAccount("t", tmp_path)
    fill = acct.book_fill("QQTMN", "BUY", 10, 50_000, 1, "maker", reason="test")
    assert math.isclose(fill["fee_tmn"], 500_000 * MAKER_FEE["TMN"])
    log = tmp_path / "t" / "decisions.jsonl"
    assert DecisionLog.verify(log) == 1
    rows = log.read_text().splitlines()
    tampered = json.loads(rows[0]); tampered["record"]["price"] = 1
    log.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValueError):
        DecisionLog.verify(log)


def engine_stub(tmp_path, trades, live_book):
    eng = Engine.__new__(Engine)
    acct = PaperAccount("a", tmp_path)
    feed = SimpleNamespace(books={"QQTMN": live_book}, usdt_tmn=lambda: 1.0,
                           trades_since=lambda s, since: [t for t in trades if t[0] > since])
    eng.feed, eng.accounts, eng.orders = feed, {"a": acct}, {"a": {}}
    return eng, acct


def test_resting_buy_fills_only_on_trades_through_price(tmp_path):
    now = time.time()
    trades = [(now - 100, 99.0, 50.0, False),   # before the order: ignored
              (now + 5, 100.0, 50.0, False),    # at our price: not counted (queue position unknown)
              (now + 6, 99.5, 3000.0, False)]   # through our price: fills 3000
    eng, acct = engine_stub(tmp_path, trades, book(99.0, 101.0))
    eng.orders["a"]["o1"] = RestingOrder("o1", "QQTMN", "BUY", 100.0, 5000.0, now, "test", {"stop": 97.0})
    eng._process_maker_fills("a", now + 10)
    pos = acct.positions()["QQTMN"]
    assert math.isclose(pos.qty, 3000.0) and pos.stop == 97.0
    assert math.isclose(eng.orders["a"]["o1"].filled, 3000.0)


def test_resting_sell_fills_when_book_crosses(tmp_path):
    now = time.time()
    eng, acct = engine_stub(tmp_path, [], book(105.0, 106.0))
    acct.book_fill("QQTMN", "BUY", 2000, 100.0, 1, "taker")
    eng.orders["a"]["o2"] = RestingOrder("o2", "QQTMN", "SELL", 104.0, 2000.0, now - 60, "tp")
    eng._process_maker_fills("a", now)
    assert acct.positions() == {} and "o2" not in eng.orders["a"]
    fill = acct.db.execute("SELECT price, liquidity FROM fills WHERE side='SELL'").fetchone()
    assert fill == (104.0, "maker")


class FakeFeed:
    def __init__(self, dev):
        self.focus = ["QQTMN", "USDTTMN"]
        self.books = {"QQTMN": book(98.0, 98.5, size=1e6)}
        self.meta = {"QQTMN": {"price_decimals": 1}}
        self.bars = {"QQTMN": Bars()}
        self._dev = dev

    def deviation(self, sym, side="mid"):
        return self._dev

    def fair(self, sym):
        return 100.0

    def premium(self, sym):
        return 0.0

    def round_price(self, sym, price, down):
        return math.floor(price * 10) / 10 if down else math.ceil(price * 10) / 10


def ctx_for(acct, feed, entries=True):
    return SimpleNamespace(feed=feed, account=acct, now=time.time(), orders=[], equity=5_000_000, entries_allowed=entries, marks={})


def test_lag_strategy_buys_only_beyond_threshold(tmp_path):
    acct = PaperAccount("l", tmp_path)
    strat = LagReversion("l", threshold=0.012, edge=0.002, stop=0.04, hold_minutes=120, size_frac=0.3, max_positions=2, max_spread=0.012)
    assert strat.decide(ctx_for(acct, FakeFeed(-0.008))) == []
    actions = strat.decide(ctx_for(acct, FakeFeed(-0.02)))
    assert [a["op"] for a in actions] == ["market_buy"] and actions[0]["symbol"] == "QQTMN"
    assert actions[0]["fields"]["stop"] < 98.5
    assert strat.decide(ctx_for(acct, FakeFeed(-0.02), entries=False)) == []


def test_lag_strategy_places_fair_value_exit(tmp_path):
    acct = PaperAccount("l", tmp_path)
    acct.book_fill("QQTMN", "BUY", 5000, 98.5, 1, "taker")
    strat = LagReversion("l", threshold=0.012, edge=0.002, stop=0.04, hold_minutes=120, size_frac=0.3, max_positions=2, max_spread=0.012)
    actions = strat.decide(ctx_for(acct, FakeFeed(0.0), entries=False))
    assert actions[0]["op"] == "place" and actions[0]["side"] == "SELL"
    assert math.isclose(actions[0]["price"], 100.2)   # fair 100 x exp(premium 0) x 1.002, rounded up to 1 decimal


def test_risk_caps_position_by_stop_distance(tmp_path):
    eng = Engine.__new__(Engine)
    acct = PaperAccount("r", tmp_path)
    feed = SimpleNamespace(books={"QQTMN": book(99.0, 100.0, size=1e9)}, usdt_tmn=lambda: 1.0)
    eng.feed, eng.accounts, eng.orders, eng.risk = feed, {"r": acct}, {"r": {}}, {"r": Risk()}
    eng.marks = lambda: {"QQTMN": 99.0}
    eng._fresh_book = lambda sym: feed.books[sym]
    ctx = SimpleNamespace(now=time.time(), entries_allowed=True)
    eng._execute("r", {"op": "market_buy", "symbol": "QQTMN", "budget_tmn": 5_000_000, "stop_pct": 0.04, "fields": {}}, ctx)
    spent = acct.db.execute("SELECT notional_tmn + fee_tmn FROM fills").fetchone()[0]
    assert spent <= 0.01 * 5_000_000 / (0.04 + 0.007) + 1     # 1% equity at risk if the stop is hit
    eng._execute("r", {"op": "market_buy", "symbol": "BADTMN".replace("BAD", "WIF"), "budget_tmn": 1_000_000, "stop_pct": 0.04, "fields": {}}, ctx)
    assert acct.db.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 1   # excluded market rejected


def test_ai_actions_are_validated_and_clamped(tmp_path):
    from experiments.wallex_arena.ai_strategy import AITrader
    key = tmp_path / "k.env"; key.write_text("ANTHROPIC_API_KEY=test-not-a-real-key\n")
    acct = PaperAccount("ai", tmp_path)
    strat = AITrader("ai", acct.dir, key, None, model="claude-haiku-5-5", mode="solo", interval_s=300, daily_usd=1, threshold=0.008)
    ctx = SimpleNamespace(feed=FakeFeed(0.0), account=acct, now=time.time(), equity=5_000_000, entries_allowed=True, log=lambda r: None)
    decision = {"actions": [
        {"symbol": "QQTMN", "action": "BUY", "size_fraction": 0.9, "stop_loss_pct": 0.5, "take_profit_pct": 0.001, "max_hold_minutes": 5, "confidence": 0.7, "reason": "x"},
        {"symbol": "NOPETMN", "action": "BUY", "size_fraction": 0.2, "stop_loss_pct": 0.02, "take_profit_pct": 0.03, "max_hold_minutes": 60, "confidence": 0.7, "reason": "y"}],
        "signal_reviews": []}
    actions = strat._translate(ctx, decision)
    assert len(actions) == 1
    a = actions[0]
    assert a["budget_tmn"] == 0.35 * 5_000_000 and a["stop_pct"] == 0.08
    assert math.isclose(a["fields"]["target"], 98.5 * 1.005) and a["fields"]["expires_at"] - ctx.now == 15 * 60


def test_ai_maker_applies_clamped_plan_and_quotes(tmp_path):
    from experiments.wallex_arena.ai_strategy import AIMaker
    key = tmp_path / "k.env"; key.write_text("ANTHROPIC_API_KEY=test-not-a-real-key\n")
    acct = PaperAccount("am", tmp_path)
    maker = {"margin": 0.004, "min_spread": 0.006, "max_spread": 0.03, "size_frac": 0.15, "max_symbols": 5,
             "max_exposure": 0.7, "stop": 0.03, "hold_minutes": 360, "min_profit": 0.001}
    strat = AIMaker("am", acct.dir, key, None, model="claude-haiku-5-5", interval_s=10**9, daily_usd=1, threshold=0.008, maker=maker)
    strat.trader.last_call = time.time()
    strat.trader.pending = {"ok": True, "decision": {"market_view": "", "notes_for_next_cycle": "n", "flatten": [],
                            "plan": [{"symbol": "QQTMN", "margin_pct": 9.0, "size_fraction": 0.9, "reason": "r"},
                                     {"symbol": "USDTTMN", "margin_pct": 0.5, "size_fraction": 0.1, "reason": "not makeable"}]}}
    logs = []
    ctx = SimpleNamespace(feed=FakeFeed(0.0), account=acct, now=time.time(), equity=5_000_000, entries_allowed=True,
                          orders=[], log=logs.append)
    actions = strat.decide(ctx)
    assert strat.maker.plan == {"QQTMN": {"margin": 0.02, "size_frac": 0.25}}
    bid = [a for a in actions if a["op"] == "place"][0]
    assert bid["side"] == "BUY" and bid["price"] <= 100.0 * (1 - 0.02) + 1e-9
    assert any(r["type"] == "maker_plan" for r in logs)


def test_usdt_rate_falls_back_to_coin_implied_rate_during_halt():
    from experiments.wallex_arena.feed import MarketFeed
    feed = MarketFeed()
    now = time.time()
    feed.meta = {s: {"base": s[:-3]} for s in ("BTCTMN", "ETHTMN", "PEPETMN", "DOGETMN")}
    feed.binance = {"BTCUSDT": (100.0, 100.0), "ETHUSDT": (10.0, 10.0), "PEPEUSDT": (1.0, 1.0), "DOGEUSDT": (2.0, 2.0)}
    def set_books(rate):
        for s, p in (("BTCTMN", 100.0), ("ETHTMN", 10.0), ("PEPETMN", 1.0), ("DOGETMN", 2.0)):
            feed.books[s] = {"mid": p * rate, "ts": time.time()}
    set_books(270_000)
    feed.books["USDTTMN"] = {"mid": 271_000.0, "ts": time.time()}
    feed.trades["USDTTMN"] = __import__("collections").deque([(now - 5, 271_000.0, 1.0, True)])
    feed._update_implied_rate(now)
    assert feed.usdt_tmn() == 271_000.0 and math.isclose(feed.rate_ratio, 271_000 / 270_000)
    feed.trades["USDTTMN"] = __import__("collections").deque([(now - 3600, 271_000.0, 1.0, True)])  # halted
    set_books(275_000)
    feed.implied.clear()
    feed._update_implied_rate(time.time())
    assert feed.usdt_halted()
    assert math.isclose(feed.usdt_tmn(), 275_000 * 271_000 / 270_000, rel_tol=1e-9)


def maker_ctx(acct, feed, liquidate_at=None, now=None):
    return SimpleNamespace(feed=feed, account=acct, now=now or time.time(), orders=[], equity=5_000_000,
                           entries_allowed=True, liquidate_at=liquidate_at, log=lambda r: None)


MAKER = {"margin": 0.004, "min_spread": 0.0, "max_spread": 0.5, "size_frac": 0.15, "max_symbols": 5, "max_exposure": 0.7,
         "stop": 0.03, "hold_minutes": 360, "min_profit": 0.001, "protections": True, "cooldown_minutes": 20,
         "max_inventories": 3, "pull_on_drop": 0.004, "relax_after_minutes": 60, "wind_down_minutes": 180}


def test_maker_protections_block_bids_and_wind_down_exits(tmp_path):
    from experiments.wallex_arena.strategies import FairValueMaker
    feed = FakeFeed(0.0); feed.meta["QQTMN"]["vol24_tmn"] = 1e10
    acct = PaperAccount("m", tmp_path)
    mk = FairValueMaker("m", **MAKER)
    bids = [a for a in mk.decide(maker_ctx(acct, feed)) if a["op"] == "place"]
    assert bids and bids[0]["side"] == "BUY"
    # global price falling fast: no bid
    b = feed.bars["QQTMN"]; m0 = int(time.time() // 60)
    for i, f in enumerate((100.0, 100.0, 99.0)):
        b.update(m0 - 2 + i, 98.0, f)
    assert not [a for a in mk.decide(maker_ctx(acct, feed)) if a["op"] == "place"]
    # wind-down: no new bids, and the exit relaxes to break-even after fees
    feed.bars["QQTMN"] = Bars()
    acct.book_fill("QQTMN", "BUY", 5000, 98.0, 1, "maker")
    acts = mk.decide(maker_ctx(acct, feed, liquidate_at=time.time() + 3600 * 2))
    sell = [a for a in acts if a["op"] == "place" and a["side"] == "SELL"][0]
    assert sell["price"] < 100.0 * 1.004 and sell["price"] >= 98.0 * 1.005 - 0.1
    assert not [a for a in acts if a["op"] == "place" and a["side"] == "BUY"]


def test_stop_loss_rests_near_fair_when_wallex_overshoots_then_falls_back_to_market(tmp_path):
    from experiments.wallex_arena.engine import Engine, Risk
    acct = PaperAccount("s", tmp_path)
    acct.book_fill("QQTMN", "BUY", 5000, 100.0, 1, "maker", position_fields={"stop": 97.0})
    crashed = book(95.0, 96.0, size=1e6)          # Wallex bid 95 while global fair value is 98
    eng = Engine.__new__(Engine)
    eng.feed = SimpleNamespace(books={"QQTMN": crashed}, fair=lambda s: 98.0, premium=lambda s: 0.0,
                               round_price=lambda s, p, down: round(p, 1), usdt_tmn=lambda: 1.0, usdt_halted=lambda: False,
                               trades_since=lambda s, since: [])
    eng.accounts, eng.orders, eng.risk = {"s": acct}, {"s": {}}, {"s": Risk()}
    eng.marks = lambda: {"QQTMN": 95.0}
    eng._fresh_book = lambda sym: crashed
    now = time.time()
    pending = eng._protective_exits("s", now)
    assert pending == {"QQTMN"} and "QQTMN" in acct.positions()      # no market sell into the wick
    (order,) = eng.orders["s"].values()
    assert order.side == "SELL" and math.isclose(order.price, 97.5)   # fair 98 minus 0.5%
    crashed["ts"] = now + 60
    assert eng._protective_exits("s", now + 60) == {"QQTMN"}
    crashed["ts"] = now + Engine.STOP_LIMIT_SECONDS + 1
    eng._protective_exits("s", now + Engine.STOP_LIMIT_SECONDS + 1)  # not filled in time: market sell
    assert "QQTMN" not in acct.positions()
    assert acct.db.execute("SELECT reason FROM fills WHERE side='SELL'").fetchone()[0] == "stop_loss_limit_expired"


def test_fast_guard_pulls_bids_and_blocks_buys_when_global_price_drops(tmp_path):
    from collections import deque
    from experiments.wallex_arena.engine import Engine, RestingOrder, Risk
    from experiments.wallex_arena.feed import MarketFeed
    feed = MarketFeed()
    now = time.time()
    feed.meta = {"QQTMN": {"base": "QQ"}}
    feed.binance = {"QQUSDT": (1.0, 1.0)}
    feed.global_hist["QQUSDT"] = deque([(now - 40, 1.000), (now - 20, 0.999), (now - 1, 0.994)])
    feed.focus = ["QQTMN"]
    feed.books = {"QQTMN": book(98.0, 99.0, size=1e6)}
    assert math.isclose(feed.global_drop("QQTMN", 60), 0.994 - 1)
    eng = Engine.__new__(Engine)
    guarded, control = PaperAccount("g", tmp_path), PaperAccount("c", tmp_path)
    eng.feed, eng.accounts = feed, {"g": guarded, "c": control}
    eng.risk = {"g": Risk(), "c": Risk(fast_guard=False)}
    eng.orders = {n: {"o" + n: RestingOrder("o" + n, "QQTMN", "BUY", 97.0, 1000.0, now - 10, "bid")} for n in ("g", "c")}
    eng.engine_log = SimpleNamespace(append=lambda r: None)
    eng._fast_guard(now)
    assert eng.orders["g"] == {} and "oc" in eng.orders["c"]
    assert eng._guarded("g", "QQTMN", now + 60) and not eng._guarded("g", "QQTMN", now + Engine.GUARD_HOLD_SECONDS + 1)


def test_ai_buys_a_market_once_per_decision(tmp_path):
    from experiments.wallex_arena.ai_strategy import AITrader
    key = tmp_path / "k.env"; key.write_text("ANTHROPIC_API_KEY=test-not-a-real-key\n")
    acct = PaperAccount("ai2", tmp_path)
    strat = AITrader("ai2", acct.dir, key, None, model="claude-haiku-5-5", mode="hybrid", interval_s=300, daily_usd=1, threshold=0.008)
    ctx = SimpleNamespace(feed=FakeFeed(-0.02), account=acct, now=time.time(), equity=5_000_000, entries_allowed=True, log=lambda r: None)
    buy = {"symbol": "QQTMN", "action": "BUY", "size_fraction": 0.1, "stop_loss_pct": 0.02, "take_profit_pct": 0.03,
           "max_hold_minutes": 60, "confidence": 0.7, "reason": "x"}
    review = {"symbol": "QQTMN", "approve": True, "size_fraction": 0.1, "reason": "y"}
    actions = strat._translate(ctx, {"actions": [buy], "signal_reviews": [review]})
    assert [a["op"] for a in actions] == ["market_buy"]


def test_maker_skips_bids_in_global_downtrend(tmp_path):
    from experiments.wallex_arena.strategies import FairValueMaker
    feed = FakeFeed(0.0); feed.meta["QQTMN"]["vol24_tmn"] = 1e10
    m0 = int(time.time() // 60) - 300
    for i in range(301):                       # fair value slides 2% over 5 hours, gently (no fast drop)
        feed.bars["QQTMN"].update(m0 + i, 98.0, 100.0 * (1 - 0.02 * i / 300))
    acct = PaperAccount("t", tmp_path)
    mk = FairValueMaker("t", **MAKER)
    assert mk._bid_blocked(maker_ctx(acct, feed), "QQTMN", {}) == "global_downtrend"
    assert not [a for a in mk.decide(maker_ctx(acct, feed)) if a["op"] == "place" and a["side"] == "BUY"]


def test_sells_below_exchange_minimum_are_rejected_like_wallex(tmp_path):
    acct = PaperAccount("d", tmp_path)
    acct.book_fill("QQTMN", "BUY", 900, 100.0, 1, "maker")              # 90,000 TMN: a partial fill below the minimum
    assert acct.market_order("QQTMN", "SELL", book(99.0, 100.0, size=1e6), 1) is None
    assert "QQTMN" in acct.positions()


def test_maker_does_not_churn_orders_on_small_jitter(tmp_path):
    import random
    from experiments.wallex_arena.engine import RestingOrder
    from experiments.wallex_arena.strategies import FairValueMaker
    rnd = random.Random(7)
    feed = FakeFeed(0.0); feed.meta["QQTMN"]["vol24_tmn"] = 1e10
    acct = PaperAccount("c", tmp_path)
    params = dict(MAKER, protections=False, min_spread=0.006, max_spread=0.03)
    mk = FairValueMaker("c", **params)
    orders, places, t0 = {}, 0, time.time()
    for i in range(60):                                   # 60 ticks of 2 s: spread wobbles around the 0.6% edge
        bid = 99.0 + rnd.choice([-0.1, 0.0, 0.1])
        feed.books["QQTMN"] = book(bid, bid * (1 + rnd.choice([0.0057, 0.0061, 0.0065])), size=1e6)
        ctx = SimpleNamespace(feed=feed, account=acct, now=t0 + 2 * i, orders=list(orders.values()), equity=5_000_000,
                              entries_allowed=True, liquidate_at=None, log=lambda r: None)
        for a in mk.decide(ctx):
            if a["op"] == "cancel":
                orders.pop(a["order_id"], None)
            elif a["op"] == "place":
                places += 1
                orders[f"o{i}"] = RestingOrder(f"o{i}", a["symbol"], a["side"], a["price"], a["qty"], t0 + 2 * i, a["reason"])
    assert places <= 4, places


def test_maker_ignores_fair_value_jitter(tmp_path):
    import random
    from experiments.wallex_arena.engine import RestingOrder
    from experiments.wallex_arena.strategies import FairValueMaker
    rnd = random.Random(11)

    class JitterFeed(FakeFeed):
        def fair(self, sym):
            return 100.0 * (1 + rnd.uniform(-0.0006, 0.0006))   # a few basis points per second, like the Binance stream

    feed = JitterFeed(0.0); feed.meta["QQTMN"]["vol24_tmn"] = 1e10
    feed.books["QQTMN"] = book(99.0, 99.8, size=1e6)
    acct = PaperAccount("j", tmp_path)
    mk = FairValueMaker("j", **dict(MAKER, protections=False, min_spread=0.006, max_spread=0.03))
    orders, places, t0 = {}, 0, time.time()
    for i in range(90):
        ctx = SimpleNamespace(feed=feed, account=acct, now=t0 + 2 * i, orders=list(orders.values()), equity=5_000_000,
                              entries_allowed=True, liquidate_at=None, log=lambda r: None)
        for a in mk.decide(ctx):
            if a["op"] == "cancel":
                orders.pop(a["order_id"], None)
            elif a["op"] == "place":
                places += 1
                orders[f"o{i}"] = RestingOrder(f"o{i}", a["symbol"], a["side"], a["price"], a["qty"], t0 + 2 * i, a["reason"])
    assert places <= 3, places
