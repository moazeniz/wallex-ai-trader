"""python -m experiments.wallex_arena run CONFIG ROOT KEYFILE | status ROOT | report ROOT [OUT.md]"""
import json
import sys
import time
from pathlib import Path


def status(root: Path):
    s = json.loads((root / "status.json").read_text())
    age = time.time() - s["updated_at"]
    print(f"updated {age:.0f}s ago | USDT/TMN {s['usdt_tmn']:,.0f} | focus {len(s['focus'])} markets | news {s.get('news', {}).get('status')}")
    print(f"{'account':18} {'equity':>11} {'ret%':>7} {'maxDD%':>7} {'fills':>5} {'closed':>6} {'wins':>5} {'fees':>8} {'open':>4} {'ord':>3}  llm$")
    for r in s["leaderboard"]:
        llm = r["strategy"].get("llm", {}).get("usd", "")
        print(f"{r['name']:18} {r['equity_tmn']:>11,} {r['return_pct']:>7.3f} {r['max_drawdown_pct']:>7.3f} {r['fills']:>5} {r['closed_trades']:>6} "
              f"{r['winning_trades']:>5} {r['fees_tmn']:>8,} {len(r['open_positions']):>4} {r['open_orders']:>3}  {llm}"
              + ("  HALTED" if r["halted"] else "") + ("  paused" if r["paused"] else ""))


def report(root: Path, out: Path | None):
    """Markdown shadow-test report built only from the arena's own logs (status, fills, hash-chained decisions)."""
    import sqlite3
    from collections import Counter
    from .account import DecisionLog
    s = json.loads((root / "status.json").read_text())
    ts = lambda x: time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(x))
    lines = [f"# Wallex arena shadow-test report", "",
             f"Generated {ts(time.time())}. Paper trading on live Wallex order books since {ts(s['started_at'])}; "
             f"every account started with 5,000,000 TMN. Fees: Wallex level-1 schedule (0.30% taker / 0.25% maker on TMN markets).", "",
             "| Account | Strategy | Equity (TMN) | Return % | Max drawdown % | Fills | Closed trades | Winning | Fees (TMN) | Volume (TMN) | LLM cost $ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["leaderboard"]:
        st = r["strategy"]
        lines.append(f"| {r['name']} | {st['class']} {st['params'].get('model', '')} | {r['equity_tmn']:,} | {r['return_pct']:.3f} | {r['max_drawdown_pct']:.3f} | "
                     f"{r['fills']} | {r['closed_trades']} | {r['winning_trades']} | {r['fees_tmn']:,} | {r['volume_tmn']:,} | {st.get('llm', {}).get('usd', '')} |")
    for r in s["leaderboard"]:
        d = root / "accounts" / r["name"]
        n = DecisionLog.verify(d / "decisions.jsonl")
        rows = [json.loads(l)["record"] for l in open(d / "decisions.jsonl")]
        kinds = Counter(x["type"] for x in rows)
        lines += ["", f"## {r['name']}", "", f"Decision log: {n} hash-chained records, chain verified. Record types: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())), ""]
        ai = [x for x in rows if x["type"] == "ai_decision"]
        if ai:
            ok = [x for x in ai if x.get("ok")]
            acts = sum(len(x["decision"].get("actions") or x["decision"].get("plan") or []) for x in ok)  # AI makers return a plan
            lines.append(f"AI calls: {len(ai)} ({len(ok)} valid), proposed actions: {acts}, "
                         f"mean latency {sum(x.get('latency_s', 0) for x in ai) / len(ai):.1f}s.")
            if ok:
                lines += ["", "Latest market view: " + ok[-1]["decision"]["market_view"][:600]]
        db = sqlite3.connect(d / "account.sqlite")
        fills = db.execute("SELECT ts, symbol, side, qty, price, notional_tmn, fee_tmn, liquidity, pnl_tmn, reason FROM fills ORDER BY id").fetchall()
        if fills:
            lines += ["", "| Time | Market | Side | Price | TMN | Fee | Liquidity | P&L (TMN) | Reason |", "|---|---|---|---:|---:|---:|---|---:|---|"]
            for f in fills:
                lines.append(f"| {ts(f[0])} | {f[1]} | {f[2]} | {f[4]:,.6g} | {f[5]:,.0f} | {f[6]:,.0f} | {f[7]} | "
                             f"{'' if f[8] is None else f'{f[8]:,.0f}'} | {f[9][:140].replace('|', '/')} |")
    text = "\n".join(lines) + "\n"
    if out:
        out.write_text(text)
    else:
        print(text)


def main():
    cmd = sys.argv[1]
    if cmd == "status":
        status(Path(sys.argv[2]))
        return
    if cmd == "report":
        report(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
        return
    if cmd == "run":
        from . import obs
        from .engine import Engine
        obs.configure("engine")
        obs.install_crash_handlers()
        obs.set_context("engine-" + obs.new_id(4))
        config = json.loads(Path(sys.argv[2]).read_text())
        obs.log("app.start", "info", "engine starting", accounts=[a["name"] for a in config["accounts"]],
                live=[a["name"] for a in config["accounts"] if a.get("live")])
        try:
            Engine(config, Path(sys.argv[3]), Path(sys.argv[4])).run()
        except BaseException as exc:
            obs.log_error("app.crash", exc, "critical")
            raise
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
