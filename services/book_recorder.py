"""Recorder v2: live /v1/depth top-of-book (+ liquidity within 0.5%) and /v1/trades for the most liquid
eligible Wallex markets, plus Binance book tickers, every PERIOD seconds. Output: SQLite at argv[1].

Usage: book_recorder.py DB_PATH [PERIOD_SECONDS]
The excluded-market list defaults to experiments/wallex_arena/excluded_markets.json (override: WALLEX_EXCLUDED_MARKETS).

/v1/markets quotes were found to be cached (top-of-book matched live depth only 9% of the time), so
execution-grade research uses depth. Universe is re-ranked by 24h TMN volume every 30 minutes.
"""
import json, os, sqlite3, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

DB = Path(sys.argv[1])
PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
N_TMN, N_USDT = 28, 14
EXCLUDED_FILE = Path(os.environ.get("WALLEX_EXCLUDED_MARKETS",
                                    Path(__file__).resolve().parent.parent / "experiments" / "wallex_arena" / "excluded_markets.json"))
EXCLUDED = set(json.loads(EXCLUDED_FILE.read_text()))


def get(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def universe():
    m = get("https://api.wallex.ir/v1/markets")["result"]["symbols"]
    usdt_tmn = num(m["USDTTMN"]["stats"]["bidPrice"]) or 1
    rows = []
    for s, v in m.items():
        if s in EXCLUDED or v["quoteAsset"] not in ("TMN", "USDT"):
            continue
        vol = num(v["stats"].get("24h_quoteVolume"))
        rows.append((s, v["quoteAsset"], vol if v["quoteAsset"] == "TMN" else vol * usdt_tmn))
    tmn = [s for s, q, _ in sorted(rows, key=lambda r: -r[2]) if q == "TMN"][:N_TMN]
    usdt = [s for s, q, _ in sorted(rows, key=lambda r: -r[2]) if q == "USDT"][:N_USDT]
    return sorted(set(tmn + usdt + ["USDTTMN"]))


def depth(sym):
    r = get(f"https://api.wallex.ir/v1/depth?symbol={sym}")["result"]
    ts = time.time()
    asks = sorted((num(x["price"]), num(x["quantity"])) for x in r["ask"])
    bids = sorted(((num(x["price"]), num(x["quantity"])) for x in r["bid"]), reverse=True)
    if not asks or not bids:
        return None
    a0, b0 = asks[0][0], bids[0][0]
    ask_liq = sum(p * q for p, q in asks if p <= a0 * 1.005)
    bid_liq = sum(p * q for p, q in bids if p >= b0 * 0.995)
    return (ts, sym, b0, a0, bids[0][1], asks[0][1], bid_liq, ask_liq)


def trades(sym):
    r = get(f"https://api.wallex.ir/v1/trades?symbol={sym}")["result"]["latestTrades"]
    out = []
    for t in r:
        ts = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00")).timestamp()
        out.append((ts, sym, num(t["price"]), num(t["quantity"]), 1 if t["isBuyOrder"] else 0))
    return out


def main():
    db = sqlite3.connect(DB)
    db.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS book(ts REAL, symbol TEXT, bid REAL, ask REAL, bid_qty REAL, ask_qty REAL, bid_liq REAL, ask_liq REAL);
    CREATE TABLE IF NOT EXISTS binance(ts REAL, symbol TEXT, bid REAL, ask REAL);
    CREATE TABLE IF NOT EXISTS trades(ts REAL, symbol TEXT, price REAL, qty REAL, buyer_taker INTEGER,
                                      UNIQUE(ts, symbol, price, qty, buyer_taker));
    CREATE TABLE IF NOT EXISTS errors(ts REAL, source TEXT, error TEXT);
    CREATE INDEX IF NOT EXISTS book_sym_ts ON book(symbol, ts);
    CREATE INDEX IF NOT EXISTS binance_sym_ts ON binance(symbol, ts);
    CREATE INDEX IF NOT EXISTS trades_sym_ts ON trades(symbol, ts);
    """)
    syms, ranked_at, cycle = [], 0, 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        started = time.time()
        if started - ranked_at > 1800:
            try:
                syms, ranked_at = universe(), started
                needed = {s[:-3] + "USDT" if s.endswith("TMN") else s for s in syms}
            except Exception as e:
                db.execute("INSERT INTO errors VALUES(?,?,?)", (started, "universe", repr(e)[:300]))
        books = list(pool.map(lambda s: _safe(depth, s), syms))
        db.executemany("INSERT INTO book VALUES(?,?,?,?,?,?,?,?)", [b for b in books if b and not isinstance(b, str)])
        for s, b in zip(syms, books):
            if isinstance(b, str):
                db.execute("INSERT INTO errors VALUES(?,?,?)", (time.time(), "depth_" + s, b))
        try:
            ts = time.time()
            db.executemany("INSERT INTO binance VALUES(?,?,?,?)", [(ts, t["symbol"], num(t["bidPrice"]), num(t["askPrice"]))
                                                                  for t in get("https://api.binance.com/api/v3/ticker/bookTicker") if t["symbol"] in needed])
        except Exception as e:
            db.execute("INSERT INTO errors VALUES(?,?,?)", (time.time(), "binance", repr(e)[:300]))
        if cycle % 2 == 0:
            for rows in pool.map(lambda s: _safe(trades, s), syms):
                if rows and not isinstance(rows, str):
                    db.executemany("INSERT OR IGNORE INTO trades VALUES(?,?,?,?,?)", rows)
        db.commit()
        cycle += 1
        time.sleep(max(0.5, PERIOD - (time.time() - started)))


def _safe(fn, sym):
    try:
        return fn(sym)
    except Exception as e:
        return repr(e)[:200]


if __name__ == "__main__":
    main()
