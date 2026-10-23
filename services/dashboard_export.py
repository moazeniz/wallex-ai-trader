"""Export arena leaderboard, equity curves (5-minute samples) and recent fills as JSON for the dashboard.

Usage: dashboard_export.py OUT_DIR [ARENA_STATE_DIR]   (state dir default: /var/lib/wallex-arena)
"""
import json, sqlite3, sys, time
from pathlib import Path
root, out = Path(sys.argv[2] if len(sys.argv) > 2 else "/var/lib/wallex-arena"), Path(sys.argv[1])
s = json.loads((root / "status.json").read_text())
APPROACH = {"LagReversion": "Lag rule", "FairValueMaker": "Maker rule", "AITrader": "AI trader", "AIMaker": "AI-managed maker", "HoldBenchmark": "Benchmark"}
board = []
for r in s["leaderboard"]:
    st = r["strategy"]; p = st["params"]
    model = (p.get("model") or "").replace("claude-", "").replace("-5-5", " 5.5").title() if p.get("model") else ""
    board.append({"account": r["name"], "approach": APPROACH.get(st["class"], st["class"]) + (" (live code, dry run)" if r["name"].startswith("live_dry") else ""),
                  "model": model or "None", "equity_tmn": r["equity_tmn"], "return_pct": round(r["return_pct"], 3),
                  "max_drawdown_pct": round(r["max_drawdown_pct"], 3), "closed_trades": r["closed_trades"], "winning_trades": r["winning_trades"],
                  "fills": r["fills"], "fees_tmn": r["fees_tmn"], "volume_tmn": r["volume_tmn"],
                  "ai_cost_usd": round(st.get("llm", {}).get("usd", 0) or 0, 2), "open_positions": len(r["open_positions"])})
equity, trades = [], []
for acct in (root / "accounts").iterdir():
    db = sqlite3.connect(acct / "account.sqlite")
    init = json.loads(db.execute("select value from meta where key=\"initial\"").fetchone()[0])
    last = 0
    for ts, eq in db.execute("select ts, equity_tmn from equity order by ts"):
        if ts - last >= 300:
            equity.append({"account": acct.name, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)), "return_pct": round(100 * (eq / init - 1), 4)})
            last = ts
    for ts, sym, side, price, notional, fee, liq, pnl, reason in db.execute("select ts,symbol,side,price,notional_tmn,fee_tmn,liquidity,pnl_tmn,reason from fills"):
        trades.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)), "account": acct.name, "market": sym, "side": side,
                       "price": price, "value_tmn": round(notional), "fee_tmn": round(fee), "liquidity": liq,
                       "pnl_tmn": None if pnl is None else round(pnl), "reason": reason[:160]})
trades.sort(key=lambda r: r["time"], reverse=True)
for i, t in enumerate(trades):
    t["id"] = "%s-%s-%d" % (t["time"], t["account"], i)
(out / "leaderboard.json").write_text(json.dumps(board))
(out / "equity.json").write_text(json.dumps(equity))
(out / "trades.json").write_text(json.dumps(trades[:60]))
print(len(board), len(equity), len(trades), s["updated_at"])
health = {"updated_at": s["updated_at"], "usdt_tmn": s.get("usdt_tmn"), "usdt_tmn_halted": s.get("usdt_tmn_halted"),
          "guarded": s.get("guarded", []), "liquidate_at_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(s["liquidate_at"])) if s.get("liquidate_at") else None,
          "wallex_stream": {k: (s.get("stream") or {}).get(k) for k in ("connected", "connections", "symbol_age_median", "reconnects")} if s.get("stream") else None,
          "binance_stream": s.get("global_stream")}
(out / "health.json").write_text(json.dumps(health))
