"""Dead-man watchdog for the live Wallex account, independent of the trading engine.

Every few seconds it checks that the engine is alive (status.json keeps being rewritten). If the engine has been
silent for `stale_after` seconds it switches to emergency mode once:
  1. cancels our resting orders (client ids starting with the engine's prefix), so no new inventory is bought and
     the coins are free;
  2. places a real STOP_MARKET sell on Wallex under every coin we hold, `stop_distance` below the current best bid,
     so the exchange itself protects the positions even if this server goes down completely.
When the engine is alive again it cancels its emergency stops and steps back; the engine's reconciliation adopts any
balance change. Orders obey the same switches as the engine (WALLEX_LIVE_ENABLED=1 and no LIVE_DISABLED file).

Run: python -m experiments.wallex_arena.watchdog STATUS_JSON STATE_JSON [--once]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path

from .account import DecisionLog
from .feed import EXCLUDED, http_json, num
from .live import ENABLE_ENV, KILL_FILE, MIN_ORDER_TMN, WallexAPIError, WallexPrivateAPI, load_markets, new_client_id

ENGINE_PREFIX = "ARENA_"
WATCHDOG_PREFIX = "WATCHDOG"


def _round(value: float, decimals: int, up: bool) -> str:
    q = Decimal(1).scaleb(-decimals)
    return str(Decimal(str(value)).quantize(q, rounding=ROUND_UP if up else ROUND_DOWN))


class Watchdog:
    def __init__(self, api: WallexPrivateAPI, markets: dict, status_path: Path, state_path: Path, *, stale_after: float = 120,
                 stop_distance: float = 0.03, kill_file: Path = KILL_FILE, env: dict | None = None, best_bid=None, clock=time.time):
        self.api, self.markets = api, markets
        self.status_path, self.state_path = Path(status_path), Path(state_path)
        self.stale_after, self.stop_distance = stale_after, stop_distance
        self.kill_file, self.env = Path(kill_file), os.environ if env is None else env
        self.best_bid = best_bid or self._public_best_bid
        self.clock = clock
        self.log = DecisionLog(self.state_path.with_suffix(".jsonl"))
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"emergency": False, "stops": []}

    @staticmethod
    def _public_best_bid(symbol: str) -> float:
        r = http_json("https://api.wallex.ir/v1/depth", {"symbol": symbol}, timeout=8)["result"]
        return max(num(x["price"]) for x in r["bid"])

    def _save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.state_path)

    def enabled(self) -> bool:
        return not self.kill_file.exists() and self.env.get(ENABLE_ENV) == "1"

    def engine_alive(self) -> bool:
        try:
            return self.clock() - self.status_path.stat().st_mtime < self.stale_after
        except OSError:
            return False

    def _holdings(self) -> dict[str, float]:
        if self.api.sub_account_id:
            raw = self.api.balances_detail() or {}
            return {a: num(b.get("total")) for a, b in raw.items()}
        raw = (self.api.balances() or {}).get("balances") or {}
        return {a: num(b.get("value")) for a, b in raw.items()}

    def tick(self) -> str:
        alive = self.engine_alive()
        if not self.enabled():
            return "disabled"
        if alive and self.state["emergency"]:
            return self._stand_down()
        if not alive and not self.state["emergency"]:
            return self._take_over()
        return "emergency" if self.state["emergency"] else "ok"

    def _take_over(self) -> str:
        now = self.clock()
        cancelled, placed, errors = [], [], []
        for o in self.api.open_orders():
            cid = str(o.get("clientOrderId") or "")
            if cid.startswith(ENGINE_PREFIX):
                try:
                    self.api.cancel_order(cid)
                    cancelled.append(cid)
                except (WallexAPIError, OSError) as exc:
                    errors.append({"cancel": cid, "error": type(exc).__name__})
        for asset, qty in self._holdings().items():
            symbol = asset + "TMN"
            m = self.markets.get(symbol)
            if asset == "TMN" or qty <= 0 or not m or symbol in EXCLUDED:
                continue
            try:
                bid = self.best_bid(symbol)
                stop = _round(bid * (1 - self.stop_distance), m["price_decimals"], up=False)
                q = _round(qty, m["qty_decimals"], up=False)
                if float(q) * float(stop) < MIN_ORDER_TMN:
                    continue  # dust the exchange would refuse
                cid = new_client_id(WATCHDOG_PREFIX)
                self.api.create_order(symbol, "SELL", "STOP_MARKET", None, q, cid, stop_price=stop)
                placed.append({"symbol": symbol, "client_id": cid, "qty": q, "stop_price": stop, "bid": bid})
            except (WallexAPIError, OSError, ValueError, KeyError) as exc:
                errors.append({"stop": symbol, "error": type(exc).__name__})
        self.state = {"emergency": True, "since": now, "stops": placed}
        self._save()
        self.log.append({"ts": now, "type": "watchdog_take_over", "cancelled": cancelled, "stops": placed, "errors": errors})
        return "took_over"

    def _stand_down(self) -> str:
        now = self.clock()
        cancelled, errors = [], []
        for s in self.state.get("stops", []):
            try:
                self.api.cancel_order(s["client_id"])
                cancelled.append(s["client_id"])
            except (WallexAPIError, OSError) as exc:  # already triggered or gone: the engine's reconciliation adopts it
                errors.append({"cancel": s["client_id"], "error": type(exc).__name__})
        self.log.append({"ts": now, "type": "watchdog_stand_down", "cancelled": cancelled, "errors": errors,
                         "emergency_seconds": round(now - self.state.get("since", now))})
        self.state = {"emergency": False, "stops": []}
        self._save()
        return "stood_down"


def main():
    from . import obs
    obs.configure("watchdog")
    obs.install_crash_handlers()
    obs.log("app.start", "info", "watchdog starting")
    status, state = Path(sys.argv[1]), Path(sys.argv[2])
    api = WallexPrivateAPI.from_env_file(Path(os.environ.get("WALLEX_ENV_FILE", "/opt/wallex/private/wallex.env")))
    dog = Watchdog(api, load_markets(), status, state)
    while True:
        try:
            dog.tick()
        except Exception as exc:  # keep watching; the next tick retries
            dog.log.append({"ts": time.time(), "type": "watchdog_error", "error": type(exc).__name__})
        if "--once" in sys.argv:
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
