"""The dead-man watchdog protects live positions with exchange-side stops when the engine goes silent."""
import os
import time

from experiments.wallex_arena.watchdog import Watchdog

MARKETS = {"QQTMN": {"base": "QQ", "quote": "TMN", "price_decimals": 0, "qty_decimals": 2, "min_qty": 0, "min_notional": 50_000},
           "DUSTTMN": {"base": "DUST", "quote": "TMN", "price_decimals": 0, "qty_decimals": 2, "min_qty": 0, "min_notional": 50_000}}


class FakeAPI:
    sub_account_id = "SUB1"

    def __init__(self):
        self.calls = []
        self.open = [{"clientOrderId": "ARENA_1_A"}, {"clientOrderId": "MANUAL_1"}]

    def open_orders(self):
        return self.open

    def cancel_order(self, cid):
        self.calls.append(("cancel", cid))
        return {}

    def balances_detail(self):
        return {"TMN": {"total": "3000000"}, "QQ": {"total": "12.3456"}, "DUST": {"total": "0.01"}}

    def create_order(self, symbol, side, type_, price, qty, cid, stop_price=None):
        self.calls.append(("create", symbol, side, type_, price, qty, cid, stop_price))
        return {}


def make(tmp_path, alive, env_on=True, kill=False):
    status = tmp_path / "status.json"; status.write_text("{}")
    if not alive:
        old = time.time() - 600
        os.utime(status, (old, old))
    kill_file = tmp_path / "LIVE_DISABLED"
    if kill:
        kill_file.write_text("")
    api = FakeAPI()
    dog = Watchdog(api, MARKETS, status, tmp_path / "watchdog.json", kill_file=kill_file,
                   env={"WALLEX_LIVE_ENABLED": "1"} if env_on else {}, best_bid=lambda s: 100_000.0)
    return dog, api, status


def test_alive_engine_means_no_action(tmp_path):
    dog, api, _ = make(tmp_path, alive=True)
    assert dog.tick() == "ok" and api.calls == []


def test_dead_engine_cancels_ours_and_places_exchange_stops(tmp_path):
    dog, api, _ = make(tmp_path, alive=False)
    assert dog.tick() == "took_over"
    assert ("cancel", "ARENA_1_A") in api.calls and ("cancel", "MANUAL_1") not in api.calls
    creates = [c for c in api.calls if c[0] == "create"]
    assert len(creates) == 1                                   # dust below the minimum order is skipped
    _, symbol, side, type_, price, qty, cid, stop = creates[0]
    assert (symbol, side, type_, price, qty, stop) == ("QQTMN", "SELL", "STOP_MARKET", None, "12.34", "97000")
    assert cid.startswith("WATCHDOG")
    assert dog.tick() == "emergency" and len([c for c in api.calls if c[0] == "create"]) == 1   # placed once


def test_recovered_engine_cancels_emergency_stops(tmp_path):
    dog, api, status = make(tmp_path, alive=False)
    dog.tick()
    stop_id = dog.state["stops"][0]["client_id"]
    status.write_text("{}")                                    # engine writes its status again
    assert dog.tick() == "stood_down"
    assert api.calls[-1] == ("cancel", stop_id) and dog.state == {"emergency": False, "stops": []}


def test_switches_off_means_no_orders(tmp_path):
    for name, kwargs in (("env_off", {"env_on": False}), ("kill_file", {"kill": True})):
        case = tmp_path / name
        case.mkdir()
        dog, api, _ = make(case, alive=False, **kwargs)
        assert dog.tick() == "disabled" and api.calls == []
