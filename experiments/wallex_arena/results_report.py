"""Backtest and shadow-test results report, built only from saved artifacts.

Inputs: the arena state directory (status, per-account ledgers, hash-chained decision logs, LLM budget ledgers),
the 1-minute backtest outputs, and the repository's git history (dates of each risk change).
Output: OUT_DIR/README.md plus PNG charts.

Usage: python -m experiments.wallex_arena.results_report ARENA_STATE BACKTEST_DIR OUT_DIR
"""
from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from .account import DecisionLog  # noqa: E402

FAMILY = {"FairValueMaker": "Maker rule", "AIMaker": "AI-managed maker", "AITrader": "AI trader",
          "LagReversion": "Lag rule", "HoldBenchmark": "Benchmark"}
FAMILY_COLOR = {"Maker rule": "#b5562d", "Maker rule + protections": "#e0a030", "AI-managed maker": "#1f6feb",
                "AI trader": "#6f42c1", "Lag rule": "#8a8f98", "Benchmark": "#2f8f5b"}
UTC = timezone.utc


def family(row: dict) -> str:
    """Strategy family from the account's described class and parameters."""
    st = row["strategy"]
    params = st.get("params", {})
    if "maker" in params:                      # AIMaker describes itself with the AITrader class name
        return "AI-managed maker"
    if st["class"] == "FairValueMaker" and params.get("protections"):
        return "Maker rule + protections"
    return FAMILY.get(st["class"], st["class"])


def ts(x: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(x))


def load_equity(acct_dir: Path) -> list[tuple[datetime, float]]:
    db = sqlite3.connect(acct_dir / "account.sqlite")
    initial = json.loads(db.execute("SELECT value FROM meta WHERE key='initial'").fetchone()[0])
    return [(datetime.fromtimestamp(t, UTC), 100 * (e / initial - 1)) for t, e in db.execute("SELECT ts, equity_tmn FROM equity ORDER BY ts")]


def equity_chart(state: Path, board: list[dict], out: Path, events: list[tuple[float, str]]):
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=130)
    seen, ends = set(), []
    for row in board:
        fam = family(row)
        pts = load_equity(state / "accounts" / row["name"])
        if not pts or row["fills"] == 0:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=FAMILY_COLOR.get(fam, "#999"), linewidth=1.4, alpha=0.9, label=None if fam in seen else fam)
        seen.add(fam)
        ends.append([ys[-1], row["name"], FAMILY_COLOR.get(fam, "#999"), xs[-1]])
    # end labels, nudged apart so they never overlap
    ends.sort()
    span = (max(e[0] for e in ends) - min(e[0] for e in ends)) or 1.0
    gap = span * 0.045
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, name, color, x in ends:
        ax.annotate(name, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7, color=color, va="center")
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    for t, label in events:
        x = datetime.fromtimestamp(t, UTC)
        ax.axvline(x, color="#bbb", linewidth=0.7)
        ax.text(x, ax.get_ylim()[1], " " + label, rotation=90, fontsize=6.5, va="top", color="#666")
    ax.set_ylabel("Net return (%) after fees")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    ax.set_xlabel("Time (UTC)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Shadow test: every account that traded, 5,000,000 TMN each", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def backtest_section(bt_dir: Path) -> list[str]:
    lines = ["## Backtest: Wallex lags the global price", ""]
    summary = bt_dir / "leadlag_summary.csv"
    if summary.exists():
        rows = list(csv.DictReader(summary.open()))
        lines.append("7 days of 1-minute Wallex candles for each eligible TMN market, Binance 1-minute closes and Wallex "
                     "USDT/TMN. Gap = ln(Wallex / (Binance x USDT/TMN)) minus the market's trailing 24 h median.")
        lines += ["", "| Horizon | Median correlation of gap with the next Wallex move | Markets |", "|---|---:|---:|"]
        for h in ("5", "15", "60", "240"):
            vals = [float(r[f"corr_dev_{h}"]) for r in rows if r.get(f"corr_dev_{h}") not in ("", "nan", None)]
            if vals:
                lines.append(f"| {h} min | {statistics.median(vals):+.3f} | {len(vals)} |")
        lines.append("")
    trades = bt_dir / "backtest_lag_TMN.csv"
    if trades.exists():
        rows = list(csv.DictReader(trades.open()))
        pooled = defaultdict(lambda: [0, 0.0, 0, 0])
        for r in rows:
            k = (r["thr"], r["edge"], r["stop"], r["hold"])
            p = pooled[k]; p[0] += int(r["n"]); p[1] += float(r["total"]); p[2] += 1; p[3] += float(r["total"]) > 0
        best = sorted(pooled.items(), key=lambda kv: -kv[1][1])[:6]
        lines += ["Market buy on a dip below fair value, limit sell back at fair value (0.30 % taker in, 0.25 % maker out, "
                  "full live spread paid on entry). Best parameter sets, all markets pooled:", "",
                  "| Gap threshold | Exit edge | Stop | Max hold | Trades | Sum of per-trade returns | Markets positive |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for (thr, edge, stop, hold), (n, total, syms, pos) in best:
            lines.append(f"| {float(thr):.1%} | {float(edge):.1%} | {float(stop):.0%} | {hold} min | {n} | {total:+.1%} | {pos}/{syms} |")
        worst = sorted(pooled.items(), key=lambda kv: kv[1][1])[0]
        lines += ["", f"The smallest gaps lose: at a {float(worst[0][0]):.1%} threshold the same rule summed to {worst[1][1]:+.1%} "
                  f"over {worst[1][0]} trades. The parameter grid was evaluated on the same 7 days, so the live shadow test "
                  "below is the out-of-sample check.", ""]
    return lines


def ai_section(state: Path, board: list[dict]) -> list[str]:
    stats = defaultdict(lambda: [0, 0, 0, 0.0])
    decisions = defaultdict(lambda: [0, 0])
    for row in board:
        budget = state / "accounts" / row["name"] / "llm-budget.sqlite"
        if budget.exists():
            for model, inp, out, usd in sqlite3.connect(budget).execute("SELECT model, input, output, usd FROM calls"):
                s = stats[model]; s[0] += 1; s[1] += inp; s[2] += out; s[3] += usd
        log = state / "accounts" / row["name"] / "decisions.jsonl"
        for line in log.open():
            r = json.loads(line)["record"]
            if r.get("type") == "ai_decision" and r.get("ok"):
                d = r["decision"]
                n = len(d.get("actions", [])) + len(d.get("plan", [])) + sum(1 for x in d.get("signal_reviews", []) if x.get("approve"))
                decisions[row["name"]][0] += 1
                decisions[row["name"]][1] += n > 0
    lines = ["## How much the AI was used", "",
             "| Model | Calls | Mean input tokens | Mean output tokens | Mean cost per decision | Total cost |",
             "|---|---:|---:|---:|---:|---:|"]
    for model, (n, i, o, usd) in sorted(stats.items()):
        lines.append(f"| {model} | {n} | {i / n:,.0f} | {o / n:,.0f} | ${usd / n:.4f} | ${usd:.2f} |")
    lines += ["", "| AI account | Valid decisions | Decisions that changed something |", "|---|---:|---:|"]
    for name, (n, acted) in sorted(decisions.items()):
        lines.append(f"| {name} | {n} | {acted} |")
    return lines + [""]


def change_events(repo: Path) -> list[tuple[float, str]]:
    keys = {"Fair-value-aware stop": "smart stop", "fast drop guard": "drop guard", "Smaller, more diversified": "smaller sizing",
            "downtrend filter": "downtrend filter", "Maker protections": "maker protections"}
    out = []
    log = subprocess.run(["git", "-C", str(repo), "log", "--format=%ct|%s"], capture_output=True, text=True).stdout
    for line in log.splitlines():
        t, subject = line.split("|", 1)
        for k, label in keys.items():
            if k.lower() in subject.lower():
                out.append((float(t), label))
    return out


def main():
    state, bt_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    status = json.loads((state / "status.json").read_text())
    board = status["leaderboard"]
    repo = Path(__file__).resolve().parents[2]
    events = [e for e in change_events(repo) if e[0] >= status["started_at"]]
    equity_chart(state, board, out_dir / "shadow-equity.png", events)

    lines = ["# Results: backtest and shadow test", "",
             f"Generated {ts(time.time())} from saved artifacts only (arena ledgers and hash-chained decision logs, backtest "
             f"outputs, git history). Shadow test running since {ts(status['started_at'])} on live Wallex order books; no "
             "real orders. Every account started with 5,000,000 TMN and pays Wallex level-1 fees.", ""]
    lines += backtest_section(bt_dir)
    lines += ["## Shadow test: all strategies on the same live data", "", "![Return of every account that traded](shadow-equity.png)", "",
              "Vertical lines mark when each risk change was deployed (from git history).", "",
              "| Account | Family | AI model | Return % | Max drawdown % | Closed trades | Profitable | Fees (TMN) |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(board, key=lambda r: -r["return_pct"]):
        fam = family(r)
        model = r["strategy"]["params"].get("model", "")
        lines.append(f"| {r['name']} | {fam} | {model} | {r['return_pct']:+.3f} | {r['max_drawdown_pct']:.3f} | {r['closed_trades']} | "
                     f"{r['winning_trades']} | {r['fees_tmn']:,} |")
    lines.append("")
    lines += ai_section(state, board)
    lines += ["## Log integrity", "", "| Account | Decision-log records | Hash chain |", "|---|---:|---|"]
    for r in board:
        n = DecisionLog.verify(state / "accounts" / r["name"] / "decisions.jsonl")
        lines.append(f"| {r['name']} | {n:,} | verified |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(out_dir / "README.md")


if __name__ == "__main__":
    main()
