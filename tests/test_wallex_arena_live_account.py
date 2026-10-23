"""LiveAccount bridges the engine to Wallex: only reported fills are booked, at the fee actually charged; resting
orders map to exchange client ids; refusals never crash the engine; reconciliation adopts exchange balances."""
import math
import time
from types import SimpleNamespace

from experiments.wallex_arena.engine import Engine, Risk
from experiments.wallex_arena.live import LiveRefused
from experiments.wallex_arena.live_account import LiveAccount

MARKETS = {"QQTMN": {"symbol": "QQTMN", "base": "QQ", "quote": "TMN", "price_decimals": 0, "qty_decimals": 4,
                     "min_qty": 0.0, "min_notional": 50_000}}


class FakeExecutor:
    def __init__(self):
        self.markets = MARKETS
        self.dry_run = False
        self.orders = {}
        self.calls = []
        self.pending_sync = []
        self.balance = {}
        self.refuse = False

    def _fill(self, cid, side, qty, price, fee, asset, liquidity="taker"):
        return {"type": "live_fill", "ts": time.time(), "client_id": cid, "symbol": "QQTMN", "side": side, "qty": qty,
                "price": price, "notional_tmn": qty * price, "fee": fee, "fee_asset": asset,
                "fee_tmn": fee * price if asset == "QQ" else fee, "liquidity": liquidity, "reason": "t"}

    def market_buy(self, symbol, budget, max_price, reason=""):
        if self.refuse:
            raise LiveRefused("WALLEX_LIVE_ENABLED is not '1'")
        self.calls.append(("market_buy", symbol, budget, max_price))
        qty = 10.0
        return {"fills": [self._fill("C1", "BUY", qty, 100_000.0, qty * 0.003, "QQ")]}   # fee taken in the coin

    def market_sell(self, symbol, qty, min_price, reason=""):
        self.calls.append(("market_sell", symbol, qty, min_price))
        return {"fills": [self._fill("C2", "SELL", qty, 101_000.0, qty * 101_000 * 0.003, "TMN")]}

    def place_limit(self, symbol, side, price, qty, reason=""):
        self.calls.append(("place_limit", symbol, side, price, qty))
        cid = f"L{len(self.calls)}"
        self.orders[cid] = SimpleNamespace(active=True, unknown=False)
        return {"order": {"client_id": cid}}

    def cancel(self, cid, reason=""):
        self.calls.append(("cancel", cid))
        self.orders[cid].active = False
        return {"fills": []}

    def sync(self):
        out, self.pending_sync = self.pending_sync, []
        return out

    def balances(self):
        return self.balance


def test_market_buy_books_reported_fill_with_fee_in_coin(tmp_path):
    ex = FakeExecutor()
    acct = LiveAccount("live", tmp_path, ex)
    book = {"asks": [(100_000.0, 100.0)], "bids": [(99_900.0, 100.0)]}
    fill = acct.market_order("QQTMN", "BUY", book, 1.0, budget_tmn=1_500_000, max_impact=0.004, position_fields={"stop": 97_000.0})
    assert ex.calls[0][0] == "market_buy" and math.isclose(ex.calls[0][3], 100_400.0)
    pos = acct.positions()["QQTMN"]
    assert math.isclose(pos.qty, 10 * (1 - 0.003))          # the coin fee reduced the holding
    assert math.isclose(acct.cash, 5_000_000 - 1_000_000)   # cash paid only the notional
    assert pos.stop == 97_000.0 and fill["live"] and fill["client_id"] == "C1"


def test_refused_order_is_logged_not_raised(tmp_path):
    ex = FakeExecutor(); ex.refuse = True
    acct = LiveAccount("live", tmp_path, ex)
    assert acct.market_order("QQTMN", "BUY", {"asks": [(1.0, 1e9)], "bids": [(1.0, 1e9)]}, 1.0, budget_tmn=1_000_000) is None
    assert acct.positions() == {}
    assert '"live_refused"' in (tmp_path / "live" / "decisions.jsonl").read_text()


def test_resting_orders_sync_fills_and_drop_inactive(tmp_path):
    ex = FakeExecutor()
    acct = LiveAccount("live", tmp_path, ex)
    eng = Engine.__new__(Engine)
    eng.feed = SimpleNamespace(books={"QQTMN": {"bid": 99_000.0, "ask": 100_000.0, "ts": time.time(),
                                                "asks": [(100_000.0, 1e9)], "bids": [(99_000.0, 1e9)]}},
                               usdt_tmn=lambda: 1.0, usdt_halted=lambda: False)
    eng.accounts, eng.orders, eng.risk = {"live": acct}, {"live": {}}, {"live": Risk()}
    eng.marks = lambda: {"QQTMN": 99_000.0}
    ctx = SimpleNamespace(now=time.time(), entries_allowed=True)
    eng._execute("live", {"op": "place", "side": "BUY", "symbol": "QQTMN", "price": 98_000.0, "qty": 5.0,
                          "fields": {"stop": 95_000.0}, "reason": "maker_bid"}, ctx)
    (cid,) = eng.orders["live"]
    assert ex.calls[-1][0] == "place_limit" and cid.startswith("L")
    ex.pending_sync = [ex._fill(cid, "BUY", 5.0, 98_000.0, 5 * 98_000 * 0.0025, "TMN", "maker")]
    ex.orders[cid].active = False
    acct.reconciled_at = time.time()
    eng._process_maker_fills("live", time.time())
    assert eng.orders["live"] == {} and math.isclose(acct.positions()["QQTMN"].qty, 5.0)
    assert acct.positions()["QQTMN"].stop == 95_000.0
    eng._execute("live", {"op": "place", "side": "SELL", "symbol": "QQTMN", "price": 99_500.0, "qty": 5.0, "reason": "ask"}, ctx)
    (cid2,) = eng.orders["live"]
    eng._execute("live", {"op": "cancel", "order_id": cid2, "reason": "requote"}, ctx)
    assert ex.calls[-1] == ("cancel", cid2) and eng.orders["live"] == {}


def test_reconcile_adopts_exchange_balances(tmp_path):
    ex = FakeExecutor()
    acct = LiveAccount("live", tmp_path, ex)
    acct.book_fill("QQTMN", "BUY", 10.0, 100_000.0, 1.0, "taker")
    ex.balance = {"TMN": {"total": 4_100_000.0}, "QQ": {"total": 9.0}}
    rec = acct.reconcile({"QQTMN": 100_000.0}, time.time(), every=0)
    assert set(rec["differences"]) == {"TMN", "QQTMN"}
    assert acct.cash == 4_100_000.0 and math.isclose(acct.positions()["QQTMN"].qty, 9.0)


def test_resting_orders_leave_rate_budget_for_protective_sells(tmp_path):
    from collections import deque
    ex = FakeExecutor()
    ex.api = SimpleNamespace(_posts=deque([time.time()] * LiveAccount.RESTING_ORDER_BUDGET))
    acct = LiveAccount("live", tmp_path, ex)
    assert acct.place_resting("QQTMN", "BUY", 98_000.0, 5.0, "maker_bid", {}) is None
    assert not any(c[0] == "place_limit" for c in ex.calls)
    assert '"resting_order_budget"' in (tmp_path / "live" / "decisions.jsonl").read_text()
    ex.api._posts.clear()
    assert acct.place_resting("QQTMN", "BUY", 98_000.0, 5.0, "maker_bid", {})
