"""Rule-based arena strategies. Each returns a list of action dicts that the engine validates and executes:

  {"op": "market_buy",  "symbol", "budget_tmn", "fields": {stop, target, expires_at, meta}, "reason", "signal"}
  {"op": "market_sell", "symbol", "qty" (None = all), "reason", "signal"}
  {"op": "place", "side": "BUY"|"SELL", "symbol", "price", "qty", "fields", "reason", "signal"}  # resting maker order
  {"op": "cancel", "order_id", "reason"}
"""
from __future__ import annotations

import math
import time

MAKER_FEE_TMN = 0.0025
MIN_ORDER_TMN = 100_000  # organizer-stated Wallex minimum order


def liquidity_budget(book: dict, side: str, max_impact: float) -> float:
    """TMN value available within max_impact of the best price."""
    levels = book["asks"] if side == "BUY" else book["bids"]
    best = levels[0][0]
    bound = best * (1 + max_impact) if side == "BUY" else best * (1 - max_impact)
    return sum(p * q for p, q in levels if (p <= bound if side == "BUY" else p >= bound))


class Strategy:
    name = "base"
    uses_maker = False

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    def decide(self, ctx) -> list[dict]:
        return []

    def describe(self) -> dict:
        return {"class": type(self).__name__, "params": self.params}


class LagReversion(Strategy):
    """Wallex TMN prices follow global prices with a delay.

    Signal: dev = ln(best ask / fair) - premium, where fair = Binance mid x Wallex USDTTMN mid and premium is the
    market's trailing 24h median of ln(mid/fair). dev < -threshold means we can buy below where the market
    normally trades relative to the global price. Buy at market; exit with a resting limit sell at the premium-
    adjusted fair value (+edge), so the exit pays the maker fee and no spread. Stop-loss and time-out exit at market.
    Backtest on 7d of 1-minute data (threshold 1.2%, edge 0.2%, hold 120 min): 53 trades, mean +0.37%/trade net.
    """

    def decide(self, ctx) -> list[dict]:
        p, feed, acct, now = self.params, ctx.feed, ctx.account, ctx.now
        actions = []
        positions = acct.positions()
        orders = {o.symbol: o for o in ctx.orders if o.side == "SELL"}
        # exits: keep one resting sell per position at the premium-adjusted fair value
        for sym, pos in positions.items():
            book, fair, prem = feed.books.get(sym), feed.fair(sym), feed.premium(sym)
            if not book or not fair or prem is None:
                continue
            target = fair * math.exp(prem) * (1 + p["edge"])
            if target <= book["bid"]:
                o = orders.get(sym)
                if o:
                    actions.append({"op": "cancel", "order_id": o.id, "reason": "target_already_bid"})
                actions.append({"op": "market_sell", "symbol": sym, "qty": None, "reason": "fair_value_reached_at_bid",
                                "signal": {"bid": book["bid"], "target": target}})
                continue
            price = feed.round_price(sym, max(target, book["bid"] * 1.0001), down=False)
            o = orders.get(sym)
            if o and abs(o.price / price - 1) < 0.001 and abs(o.qty - pos.qty) / pos.qty < 0.01:
                continue
            if o:
                actions.append({"op": "cancel", "order_id": o.id, "reason": "reprice_exit"})
            actions.append({"op": "place", "side": "SELL", "symbol": sym, "price": price, "qty": pos.qty,
                            "reason": "exit_at_fair_value", "signal": {"fair": fair, "premium": prem, "target": target}})
        if not ctx.entries_allowed:
            return actions
        # entries: biggest discount first
        slots = p["max_positions"] - len(positions)
        candidates = []
        for sym in feed.focus:
            if sym == "USDTTMN" or sym in positions:
                continue
            book = feed.books.get(sym)
            if not book or now - book["ts"] > 15 or book["spread"] > p["max_spread"]:
                continue
            if feed.meta.get(sym, {}).get("vol24_tmn", 0) < p.get("min_vol24_tmn", 0):
                continue  # the backtested edge was in the ~20 busiest markets; thin books gap and stay gapped
            dev = feed.deviation(sym, "ask")
            if dev is None or dev > -p["threshold"]:
                continue
            candidates.append((dev, sym, book))
        candidates.sort()
        equity = ctx.equity
        for dev, sym, book in candidates[: max(0, slots)]:
            budget = min(p["size_frac"] * equity, liquidity_budget(book, "BUY", 0.004))
            actions.append({"op": "market_buy", "symbol": sym, "budget_tmn": budget,
                            "fields": {"stop": book["ask"] * (1 - p["stop"]), "expires_at": now + p["hold_minutes"] * 60},
                            "reason": f"wallex_ask_{dev:.2%}_below_global_fair",
                            "signal": {"dev_ask": dev, "fair": feed.fair(sym), "premium": feed.premium(sym), "ask": book["ask"],
                                       "spread": book["spread"], "binance_15m": feed.bars[sym].fair_change(15) if sym in feed.bars else None}})
        return actions


class FairValueMaker(Strategy):
    """Provide liquidity on Wallex around the global fair value.

    Rest a bid just inside the spread but never above fair*(1-margin); after a fill, rest an ask never below
    fair*(1+margin) nor below entry + fees. The global price anchors quotes, which limits adverse selection when
    Wallex lags Binance. Fills are only counted when real Wallex trades print through our price (or the book crosses it).

    `plan` (set by an AI manager, see ai_strategy.AIMaker) overrides market selection: {symbol: {"margin", "size_frac"}}.
    """
    uses_maker = True
    plan: dict | None = None

    def _param(self, sym, key):
        if self.plan is not None and sym in self.plan and key in self.plan[sym]:
            return self.plan[sym][key]
        return self.params[key]

    def decide(self, ctx) -> list[dict]:
        p, feed, acct, now = self.params, ctx.feed, ctx.account, ctx.now
        actions = []
        positions = acct.positions()
        by_key = {(o.symbol, o.side): o for o in ctx.orders}
        equity = ctx.equity
        exposure = sum(pos.qty * feed.books.get(s, {}).get("bid", pos.entry_price) for s, pos in positions.items())
        # choose markets: wide enough spread to earn > 2 maker fees, liquid, with a global reference
        ranked = []
        for sym in feed.focus:
            book = feed.books.get(sym)
            if sym == "USDTTMN" or not book or now - book["ts"] > 15:
                continue
            if feed.fair(sym) is None or feed.premium(sym) is None:
                continue
            if p["min_spread"] <= book["spread"] <= p["max_spread"]:
                ranked.append((-(feed.meta.get(sym, {}).get("vol24_tmn", 0)), sym))
        if self.plan is not None:
            chosen = {s for s in self.plan if s in feed.focus and feed.books.get(s) and feed.fair(s) and feed.premium(s) is not None}
        elif p.get("symbols"):
            # a fixed market list (e.g. the USDC stablecoin market), quoted whatever its rank by volume
            chosen = {s for s in p["symbols"] if feed.books.get(s) and feed.fair(s) and feed.premium(s) is not None}
        else:
            chosen = self._sticky_selection(feed, ranked, now)
        active = chosen | set(positions)
        for sym in sorted(active):
            book = feed.books.get(sym)
            fair = feed.fair_smooth(sym) if hasattr(feed, "fair_smooth") else feed.fair(sym)
            prem = feed.premium(sym)
            if not book or not fair or prem is None:
                continue
            anchor = fair * math.exp(prem)
            tick = 10 ** -feed.meta.get(sym, {}).get("price_decimals", 0)
            pos = positions.get(sym)
            if pos and pos.qty * book["bid"] < MIN_ORDER_TMN * 1.02:
                pos = None  # below the exchange minimum it cannot be sold: keep bidding to top it up instead
            if pos:
                protect = p.get("protections", False)
                to_end = (ctx.liquidate_at - now) if protect and getattr(ctx, "liquidate_at", None) else float("inf")
                wind_down = p.get("wind_down_minutes", 180) * 60
                aged = protect and now - pos.opened_at > p.get("relax_after_minutes", 60) * 60
                if to_end < wind_down / 3:
                    # final hour of the wind-down: sell just inside the spread, accepting a small loss over a 0.3% taker exit
                    want = max(book["ask"] - tick, pos.entry_price * (1 - 0.003))
                elif to_end < wind_down or aged:
                    # wind-down or stale inventory: give up the margin, keep break-even after both maker fees
                    want = max(book["ask"] - tick, anchor, pos.entry_price * (1 + 2 * MAKER_FEE_TMN))
                else:
                    floor_price = pos.entry_price * (1 + 2 * MAKER_FEE_TMN + p["min_profit"])
                    want = max(book["ask"] - tick, anchor * (1 + self._param(sym, "margin")), floor_price)
                want = feed.round_price(sym, max(want, book["bid"] + tick), down=False)
                self._quote(actions, by_key.get((sym, "SELL")), sym, "SELL", want, pos.qty, "maker_ask", {"anchor": anchor},
                            now=now, tick=tick)
                stale = by_key.get((sym, "BUY"))
                if stale:
                    actions.append({"op": "cancel", "order_id": stale.id, "reason": "holding_inventory"})
                continue
            blocked = self._bid_blocked(ctx, sym, positions)
            if not ctx.entries_allowed or sym not in chosen or blocked:
                o = by_key.get((sym, "BUY"))
                if o:
                    actions.append({"op": "cancel", "order_id": o.id, "reason": blocked or "market_deselected"})
                continue
            want = min(book["bid"] + tick, anchor * (1 - self._param(sym, "margin")))
            want = feed.round_price(sym, want, down=True)
            size_frac = self._param(sym, "size_frac")
            if want >= book["ask"] or exposure + size_frac * equity > p["max_exposure"] * equity:
                o = by_key.get((sym, "BUY"))
                if o:
                    actions.append({"op": "cancel", "order_id": o.id, "reason": "no_room"})
                continue
            qty = size_frac * equity / want
            self._quote(actions, by_key.get((sym, "BUY")), sym, "BUY", want, qty, "maker_bid",
                        {"anchor": anchor, "fair": fair, "premium": prem, "spread": book["spread"]},
                        fields={"stop": want * (1 - p["stop"]), "expires_at": now + p["hold_minutes"] * 60},
                        now=now, tick=tick, ceiling=anchor * (1 - self._param(sym, "margin")))
        for (sym, side), o in by_key.items():
            if sym not in active:
                actions.append({"op": "cancel", "order_id": o.id, "reason": "market_deselected"})
        return actions

    def _bid_blocked(self, ctx, sym, positions) -> str | None:
        """Reasons not to rest a new bid now (protections against accumulating inventory in a falling market)."""
        p, now = self.params, ctx.now
        if not p.get("protections", False):
            return None
        if getattr(ctx, "liquidate_at", None) and ctx.liquidate_at - now < p.get("wind_down_minutes", 180) * 60:
            return "wind_down_no_new_inventory"
        if len(positions) >= p.get("max_inventories", 3):
            return "inventory_cap"
        bars = ctx.feed.bars.get(sym)
        drop = bars.fair_change(2) if bars else None
        if drop is not None and drop < -p.get("pull_on_drop", 0.004):
            return "global_price_falling_fast"
        # A maker's bids fill most often in coins that keep falling (FIL, PUMP on 2026-10-22): no new inventory in a
        # coin whose global price is in a downtrend.
        trend4, trend12 = (bars.fair_change(240), bars.fair_change(720)) if bars else (None, None)
        held = getattr(self, "_downtrend_until", {})
        if (trend4 is not None and trend4 < -p.get("trend_4h_limit", 0.01)) or (trend12 is not None and trend12 < -p.get("trend_12h_limit", 0.03)):
            held[sym] = now + 600   # hold the block 10 minutes so a trend near the limit does not flip it every tick
            self._downtrend_until = held
        if now < held.get(sym, 0):
            return "global_downtrend"
        last_buy = ctx.account.db.execute("SELECT MAX(ts) FROM fills WHERE symbol=? AND side='BUY'", (sym,)).fetchone()[0]
        if last_buy and now - last_buy < p.get("cooldown_minutes", 20) * 60:
            return "cooldown_after_fill"
        return None

    MIN_QUOTE_LIFE = 45        # seconds before a quote may be moved for a small price change
    BIG_MOVE = 0.005           # a move this large requotes at once
    KEEP_SELECTED = 600        # a selected market stays selected at least this long (seconds)

    def _sticky_selection(self, feed, ranked, now) -> set:
        """Markets to quote, with hysteresis: a market near the spread band's edge must not flip in and out every tick
        (each flip cancels and re-places the bid, losing its place in the queue)."""
        p = self.params
        selected = getattr(self, "_selected", {})
        keep = {}
        for sym, since in selected.items():
            book = feed.books.get(sym)
            if not book or feed.fair(sym) is None or feed.premium(sym) is None:
                continue
            in_band = 0.8 * p["min_spread"] <= book["spread"] <= 1.2 * p["max_spread"]
            if in_band or now - since < self.KEEP_SELECTED:
                keep[sym] = since
        for _, sym in sorted(ranked):
            if len(keep) >= p["max_symbols"]:
                break
            keep.setdefault(sym, now)
        self._selected = keep
        return set(keep)

    def _quote(self, actions, existing, sym, side, price, qty, reason, signal, fields=None, now=None, tick=0.0, ceiling=None):
        """Place or move a resting order. Small moves wait MIN_QUOTE_LIFE so the order keeps its queue position; a bid
        now above the fair-value ceiling (the global price fell) is moved at once."""
        if existing:
            moved = abs(existing.price / price - 1)
            tolerance = max(0.0015, 2 * tick / price) if price else 0.0015
            # urgent only when the bid is clearly above the fair-value ceiling, not on a few basis points of jitter
            above_ceiling = side == "BUY" and ceiling is not None and existing.price > ceiling * (1 + tolerance)
            age = (now - existing.placed_at) if now is not None else float("inf")
            if not above_ceiling and (moved < tolerance or (age < self.MIN_QUOTE_LIFE and moved < self.BIG_MOVE)):
                return
        if existing:
            actions.append({"op": "cancel", "order_id": existing.id, "reason": "requote"})
        actions.append({"op": "place", "side": side, "symbol": sym, "price": price, "qty": qty, "reason": reason,
                        "signal": signal, "fields": fields or {}})


class HoldBenchmark(Strategy):
    """Buy one asset once at start and hold (benchmark for the demo; liquidated at the cutoff like everything)."""

    def decide(self, ctx) -> list[dict]:
        sym = self.params["symbol"]
        if ctx.account.positions() or not ctx.entries_allowed or ctx.account.db.execute("SELECT COUNT(*) FROM fills").fetchone()[0]:
            return []
        book = ctx.feed.books.get(sym)
        if not book:
            return []
        return [{"op": "market_buy", "symbol": sym, "budget_tmn": ctx.equity * self.params["fraction"],
                 "fields": {}, "reason": "benchmark_buy_and_hold", "signal": {}}]
