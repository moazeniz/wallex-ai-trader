"""Password-protected dashboard for meowmeow-wallex on Liara (stdlib only).

/                 arena dashboard over the Hetzner PAPER/SHADOW snapshot (banner with snapshot time)
/live             live engine control: ARMED/DISARMED, engine process, equity, positions, realized PnL, decisions, errors
/report           shadow-test report (all accounts, every fill with its reason, hash-chain verification)
/decisions        decision-log viewer (?src=snapshot|live&account=NAME)
/paper-agent      Paper agent snapshot (portfolio, decisions)
/docs/NAME        Wallex docs and the 7-day backtest table
/files/PATH       raw snapshot files (hash-chained decisions.jsonl, account.sqlite ...)
/api/live, POST /api/disarm, POST /api/arm, POST /api/balances (read-only)
/healthz          unauthenticated liveness
Basic auth: user "admin", password WALLEX_ADMIN_PASSWORD. Without that variable every page except /healthz is refused.
"""
import base64
import calendar
import threading
import hmac
import html
import json
import os
import sqlite3
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
from experiments.wallex_arena import obs  # noqa: E402
SNAP = APP / "snapshot"
LIVE = Path(os.environ.get("WALLEX_STATE_DIR") or "/var/lib/wallex-arena")
KILL = LIVE / "LIVE_DISABLED"
RUN = Path(os.environ.get("WALLEX_RUN_DIR") or ("/run/wallex" if Path("/run/wallex").exists() else "/tmp/wallex"))
USER = "admin"
CONFIRM = "ARM LIVE TRADING"
CT = {".json": "application/json", ".jsonl": "application/x-ndjson", ".md": "text/markdown; charset=utf-8", ".txt": "text/plain; charset=utf-8",
      ".sqlite": "application/vnd.sqlite3", ".html": "text/html; charset=utf-8"}

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16191d;--mut:#5d6672;--line:#dfe3e8;--acc:#1f5fbf;--ok:#16794a;--bad:#b42318;--warn:#a15c07}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#181c21;--ink:#e8ebef;--mut:#9aa4b0;--line:#2a3038;--acc:#6ea8ff;--ok:#4cc38a;--bad:#ff7b72;--warn:#e3a33b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--card)}
html,body{overflow-x:hidden}.wrap{overflow-x:auto;max-width:100%}.card{overflow-wrap:anywhere}
@media (max-width:760px){main{padding:10px}table{display:block;overflow-x:auto;max-width:100%}.v{font-size:17px}}
header b{font-size:15px}header a{color:var(--acc);text-decoration:none}main{max-width:1200px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.k{color:var(--mut);font-size:12px}.v{font-size:20px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:500}td.n{text-align:right;font-variant-numeric:tabular-nums}.pos{color:var(--ok)}.neg{color:var(--bad)}
.pill{display:inline-block;padding:3px 10px;border-radius:99px;font-weight:700;letter-spacing:.03em}.armed{background:var(--bad);color:#fff}.disarmed{background:var(--ok);color:#fff}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.45 ui-monospace,Menlo,monospace}.wrap{overflow-x:auto}button{font:inherit;padding:6px 14px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.danger{border-color:var(--bad);color:var(--bad)}input{font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.banner{background:#fff4d6;color:#4a3500;border-bottom:1px solid #e8cf86;padding:9px 20px;font:13px/1.45 system-ui,sans-serif}
@media (prefers-color-scheme:dark){.banner{background:#3a2e0b;color:#f4e2a8;border-color:#5b4815}}
"""
NAV = ('<a href="/">Arena results</a><a href="/live">Live engine</a><a href="/market">Market (MCP)</a><a href="/report">Shadow-test report</a>'
       '<a href="/decisions">Decision logs</a><a href="/paper-agent">Paper agent</a><a href="/docs">Docs &amp; backtest</a>')
NAV_PATHS = ["/", "/live", "/market", "/report", "/decisions", "/paper-agent", "/docs"]
NAV_INLINE = NAV.replace("<a ", '<a style="margin-right:12px;display:inline-block" ')
# Injected into the arena page so it never scrolls sideways on phones: tables/charts scroll inside their own box.
MOBILE_CSS = ("<style>.mm-banner a{color:#1f5fbf}@media (max-width:760px){html,body{overflow-x:hidden}"
              "table{display:block;max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch}"
              "svg,canvas,img{max-width:100%;height:auto}pre{white-space:pre-wrap;word-break:break-word}}</style>")


PUSHED = Path(os.environ.get("WALLEX_SNAPSHOT_DIR") or (LIVE.parent / "snapshots"))   # /data/snapshots on Liara
PUSH_MAX_BYTES = int(os.environ.get("SNAPSHOT_PUSH_MAX_BYTES") or 25_000_000)          # compressed request body
PUSH_MAX_UNPACKED = 150_000_000
UPDATED_FA = "آخرین به‌روزرسانی"
SHADOW_FA = "اختبار سایه/کاغذی"


def pushed_dir() -> Path | None:
    cur = PUSHED / "current"
    return cur if (cur / "MANIFEST.json").is_file() else None


def sp(rel: str) -> Path:
    """A snapshot path: from the newest pushed snapshot when it has it, else from the snapshot bundled in the image."""
    cur = pushed_dir()
    if cur is not None and (cur / rel).exists():
        return cur / rel
    return SNAP / rel


def manifest() -> dict:
    for base in filter(None, (pushed_dir(), SNAP)):
        try:
            m = json.loads((base / "MANIFEST.json").read_text())
            m["_source"] = "pushed" if base != SNAP else "bundled"
            return m
        except (OSError, ValueError):
            continue
    return {}


def snapshot_info() -> dict:
    m = manifest()
    ts = m.get("snapshot_ts")
    return {"source": m.get("_source"), "snapshot_utc": m.get("snapshot_utc"), "snapshot_ts": ts,
            "age_s": round(time.time() - ts) if ts else None, "pushed_at": m.get("pushed_at"), "kind": m.get("kind", "full"),
            "arena_started_utc": m.get("arena_started_utc"), "files": len(m.get("files") or {})}


def updated_line(m: dict) -> str:
    """Prominent Persian 'last update' line (Tehran and UTC); the page's script refreshes it every 60 s."""
    ts = m.get("snapshot_ts")
    when = (time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts + 12600)) + " تهران / " + time.strftime("%H:%M UTC", time.gmtime(ts))) if ts else "?"
    return (f'<div dir="rtl" style="font-size:16px;font-weight:700;margin:2px 0 4px">{UPDATED_FA}: <span id="mm-upd">{html.escape(when)}</span> '
            f'({SHADOW_FA}) <span id="mm-age" style="font-weight:400;font-size:13px"></span></div>')


UPDATE_JS = """<script>(function(){const f=n=>String(n).padStart(2,'0');
function fmt(ts){const t=new Date((ts+12600)*1000),u=new Date(ts*1000);return t.getUTCFullYear()+'-'+f(t.getUTCMonth()+1)+'-'+f(t.getUTCDate())+' '+f(t.getUTCHours())+':'+f(t.getUTCMinutes())+' تهران / '+f(u.getUTCHours())+':'+f(u.getUTCMinutes())+' UTC'}
async function tick(){try{const d=await (await fetch('/api/snapshot/info',{cache:'no-store'})).json();if(!d.snapshot_ts)return;
const e=document.getElementById('mm-upd');if(e)e.textContent=fmt(d.snapshot_ts);const a=document.getElementById('mm-age');
if(a)a.textContent='('+Math.max(0,Math.round(d.age_s/60))+' دقیقه پیش، '+(d.source==='pushed'?'به‌روزرسانی خودکار':'نسخهٔ همراه برنامه')+')';}catch(e){}}
tick();setInterval(tick,60000);})();</script>"""


def page(title, body):
    m = manifest()
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body><header><b>MeowMeow Wallex</b>{NAV}</header>'
            f'<div class="banner">{updated_line(m)}<b>Paper / shadow trading results.</b> Snapshot of the Hetzner arena taken {html.escape(m.get("snapshot_utc", "?"))} '
            f'(arena running since {html.escape(m.get("arena_started_utc", "?"))}); simulated 5,000,000 TMN per account on live Wallex order books, no real orders. '
            f'Live engine: see <a href="/live">Live engine</a>.</div><main>{body}</main>{UPDATE_JS}</body></html>').encode()


def fmt(x, d=0):
    try:
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return "" if x is None else html.escape(str(x))


def tail_jsonl(path: Path, n=60, max_bytes=3_000_000):
    if not path.is_file():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        lines = f.read().splitlines()[1 if size > max_bytes else 0:]
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line).get("record", {}))
        except ValueError:
            pass
    return out


def summarize(r: dict) -> str:
    t = r.get("type", "")
    if t == "ai_decision":
        if not r.get("ok"):
            return f"AI call failed: {r.get('error', '')}"
        d = r.get("decision") or {}
        parts = [d.get("market_view", "")]
        for p in d.get("plan", []) or []:
            parts.append(f"QUOTE {p.get('symbol')} margin {p.get('margin_pct')}% size {p.get('size_fraction')}: {p.get('reason', '')}")
        for a in d.get("actions", []) or []:
            parts.append(f"{a.get('action')} {a.get('symbol')} size {a.get('size_fraction')}: {a.get('reason', '')}")
        if d.get("flatten"):
            parts.append("FLATTEN " + ", ".join(d["flatten"]))
        return " | ".join(x for x in parts if x) + f"  [{r.get('model', '')}, ${fmt(r.get('usd'), 4)}]"
    if t == "fill" or "side" in r and "symbol" in r and "price" in r:
        return f"{r.get('side')} {r.get('symbol')} {fmt(r.get('notional_tmn'))} TMN @ {fmt(r.get('price'), 2)} pnl {fmt(r.get('pnl_tmn'))} | {r.get('reason', '')}"
    keep = {k: v for k, v in r.items() if k not in ("ts", "type", "input") and len(json.dumps(v, default=str)) < 400}
    return json.dumps(keep, ensure_ascii=False, default=str)[:600]


def decisions_table(records):
    rows = []
    for r in reversed(records):
        ts = time.strftime("%m-%d %H:%M:%S", time.gmtime(r["ts"])) if isinstance(r.get("ts"), (int, float)) else ""
        rows.append(f"<tr><td>{ts}</td><td>{html.escape(str(r.get('type', '')))}</td><td>{html.escape(summarize(r))}</td></tr>")
    return '<div class="wrap"><table><tr><th>UTC</th><th>Type</th><th>Decision / event and reason</th></tr>' + "".join(rows) + "</table></div>"


def account_rows(root: Path):
    try:
        s = json.loads((root / "status.json").read_text())
    except (OSError, ValueError):
        return None, []
    rows = []
    for r in s.get("leaderboard", []):
        realized = None
        db = root / "accounts" / r["name"] / "account.sqlite"
        if db.is_file():
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
                    realized = c.execute("SELECT COALESCE(SUM(pnl_tmn),0) FROM fills WHERE pnl_tmn IS NOT NULL").fetchone()[0]
            except sqlite3.Error:
                pass
        rows.append({**r, "realized_pnl_tmn": realized})
    return s, rows


def armed_state() -> dict:
    env = os.environ.get("WALLEX_LIVE_ENABLED") == "1"
    key = bool(os.environ.get("WALLEX_API_KEY"))
    kill = KILL.exists()
    try:
        proc = json.loads((RUN / "engine-proc.json").read_text())
    except (OSError, ValueError):
        proc = {}
    engine_env = (os.environ.get("WALLEX_ENGINE") or "off").lower() in ("on", "1", "true", "yes")
    return {"armed": env and key and not kill and engine_env, "env_live_enabled": env, "api_key_set": key,
            "subaccount_set": bool(os.environ.get("WALLEX_SUBACCOUNT_ID")), "kill_file_present": kill,
            "engine_enabled": engine_env, "engine_process": proc}


def challenge_info(status: dict | None) -> dict:
    """Countdown and settlement status: from the running engine when there is one, else from the generated config."""
    now = time.time()
    try:
        cfg = json.loads((RUN / "live-config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    ch = (status or {}).get("challenge") or {}
    parse = lambda s: calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) if s else None
    end = ch.get("end") or parse(cfg.get("challenge_end_utc"))
    flat_start = ch.get("flatten_start") or (end - 60 * float(cfg.get("flatten_minutes", 60)) if end else None)
    stop = ch.get("entries_stop") or (end - 60 * float(cfg.get("no_new_entries_minutes", 120)) if end else None)
    phase = ch.get("phase") or ("no_end_configured" if not end else "ended" if now >= end else "flattening" if now >= flat_start
                                else "no_new_entries" if now > stop else "trading")
    iso = lambda x: time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(x)) if x else None
    return {"end_utc": iso(end), "flatten_from_utc": iso(flat_start), "no_new_entries_from_utc": iso(stop), "phase": phase,
            "seconds_to_end": round(end - now) if end else None, "seconds_to_flatten": round(flat_start - now) if flat_start else None,
            "flat": ch.get("flat"), "engine_reporting": bool(ch), "placeholder_end": cfg.get("challenge_end_is_placeholder", True),
            "max_exposure": ((cfg.get("accounts") or [{}])[0].get("params", {}).get("maker", {}) or {}).get("max_exposure")}


def api_live() -> dict:
    s, rows = account_rows(LIVE)
    try:
        check = json.loads((RUN / "selfcheck.json").read_text())
    except (OSError, ValueError):
        check = None
    out = {"now": time.time(), "state": armed_state(), "selfcheck": check, "status": None, "accounts": [], "decisions": [], "errors": [],
           "challenge": challenge_info(s)}
    if s:
        out["status"] = {k: s.get(k) for k in ("updated_at", "started_at", "usdt_tmn", "usdt_tmn_halted", "guarded", "liquidate_at", "stream", "global_stream", "news")}
        out["errors"] = s.get("feed_errors_recent", [])[-20:]
        out["accounts"] = [{k: r.get(k) for k in ("name", "equity_tmn", "return_pct", "max_drawdown_pct", "fills", "closed_trades", "winning_trades",
                                                  "fees_tmn", "open_positions", "open_orders", "halted", "paused", "realized_pnl_tmn")} for r in rows]
        for r in rows:
            for rec in tail_jsonl(LIVE / "accounts" / r["name"] / "decisions.jsonl", 40):
                out["decisions"].append({"account": r["name"], "ts": rec.get("ts"), "type": rec.get("type"), "summary": summarize(rec)})
        out["decisions"].sort(key=lambda d: d.get("ts") or 0, reverse=True)
        out["decisions"] = out["decisions"][:80]
    for f in ("engine.jsonl", "control.jsonl"):
        for rec in tail_jsonl(LIVE / f, 10):
            if "error" in json.dumps(rec).lower():
                out["errors"].append(rec)
    return out


_MCP = None
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def mcp_client():
    global _MCP
    if _MCP is None:
        from experiments.wallex_arena.wallex_mcp import WallexMCP
        _MCP = WallexMCP(log_path=os.environ.get("WALLEX_MCP_LOG") or (LIVE / "mcp_calls.jsonl"))
    return _MCP


def cached(key, ttl, fn):
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)
    return value


def mcp_calls(n=100) -> list[dict]:
    path = Path(os.environ.get("WALLEX_MCP_LOG") or (LIVE / "mcp_calls.jsonl"))
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines()[-n:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows[::-1]


MARKET_PAGE = """
<div class="card"><b>Wallex market data via MCP</b>
<p class="k">Served by the official Wallex MCP server (<code>https://api.wallex.ir/mcp</code>, public read-only tools, no API key);
when it is unavailable the same data comes from Wallex public REST and the source column says <code>rest</code>.
The AI uses the same tools in its research step before each decision.</p>
<p><select id="sym"></select> <select id="res"><option value="15">15 min</option><option value="60" selected>1 hour</option>
<option value="240">4 hours</option><option value="D">1 day</option></select>
<select id="hrs"><option value="24">24 h</option><option value="72" selected>3 days</option><option value="168">7 days</option><option value="720">30 days</option></select>
<button onclick="draw()">Load</button> <span id="src" class="k"></span></p>
<div class="wrap"><svg id="chart" viewBox="0 0 900 360" style="width:100%;height:auto;background:var(--bg);border-radius:8px"></svg></div>
<div class="grid" id="stats" style="margin-top:10px"></div></div>
<div class="card"><b>Recent MCP calls</b> <span class="k">(engine and dashboard; newest first)</span><div class="wrap" id="calls"></div></div>
<div class="card"><b>Tools on the Wallex MCP server</b> <span class="k">(only the allow-listed public market-data tools can be called)</span><div class="wrap" id="tools"></div></div>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const SYMS=['BTCTMN','ETHTMN','USDTTMN','SUITMN','FILTMN','DOGETMN','XRPTMN','SOLTMN','ARBTMN','PEPETMN','TRXTMN','BNBTMN'];
$('sym').innerHTML=SYMS.map(s=>`<option>${s}</option>`).join('');
async function draw(){const r=await fetch(`/api/market/candles?symbol=${$('sym').value}&resolution=${$('res').value}&hours=${$('hrs').value}`);const d=await r.json();
 const svg=$('chart'); if(!r.ok||!d.bars||!d.bars.length){svg.innerHTML='<text x="20" y="40" fill="currentColor">'+esc(d.error||'no data')+'</text>';return}
 $('src').textContent=`source: ${d.source} | ${d.bars.length} bars | change ${d.change_pct}%`;
 const W=900,H=360,P=50,b=d.bars,hi=Math.max(...b.map(x=>x[2])),lo=Math.min(...b.map(x=>x[3])),n=b.length,cw=(W-P-10)/n;
 const y=v=>10+(H-40)*(hi-v)/((hi-lo)||1); let g='';
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;g+=`<line x1="${P}" x2="${W-5}" y1="${y(v)}" y2="${y(v)}" stroke="currentColor" opacity=".12"/><text x="2" y="${y(v)+4}" font-size="10" fill="currentColor" opacity=".7">${Number(v).toLocaleString('en-US',{maximumSignificantDigits:6})}</text>`}
 b.forEach((x,i)=>{const cx=P+i*cw+cw/2,up=x[4]>=x[1],col=up?'#16794a':'#b42318';
  g+=`<line x1="${cx}" x2="${cx}" y1="${y(x[2])}" y2="${y(x[3])}" stroke="${col}"/><rect x="${cx-Math.max(1,cw*.35)}" width="${Math.max(2,cw*.7)}" y="${Math.min(y(x[1]),y(x[4]))}" height="${Math.max(1,Math.abs(y(x[1])-y(x[4])))}" fill="${col}"><title>${new Date(x[0]*1000).toISOString().slice(0,16)} O ${x[1]} H ${x[2]} L ${x[3]} C ${x[4]} V ${x[5]}</title></rect>`});
 const t0=new Date(b[0][0]*1000).toISOString().slice(5,16).replace('T',' '),t1=new Date(b[n-1][0]*1000).toISOString().slice(5,16).replace('T',' ');
 g+=`<text x="${P}" y="${H-8}" font-size="11" fill="currentColor" opacity=".7">${t0} UTC</text><text x="${W-120}" y="${H-8}" font-size="11" fill="currentColor" opacity=".7">${t1} UTC</text>`;
 svg.innerHTML=g; const ob=await (await fetch('/api/market/book?symbol='+$('sym').value)).json();
 const tiles=[['Last close',d.last_close],['High',d.high],['Low',d.low],['Best bid',ob.best_bid],['Best ask',ob.best_ask],['Spread %',ob.spread_pct],['Bid TMN within 1%',ob.bid_tmn_within_1pct],['Book source',ob.source]];
 $('stats').innerHTML=tiles.map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${esc(typeof v==='number'?v.toLocaleString('en-US'):v)}</div></div>`).join(''); calls()}
async function calls(){const c=await (await fetch('/api/mcp/calls')).json();
 $('calls').innerHTML=c.length?'<table><tr><th>UTC</th><th>Tool</th><th>Args</th><th>Status</th><th>Source</th><th>ms</th></tr>'+c.slice(0,60).map(x=>`<tr><td>${new Date(x.ts*1000).toISOString().slice(5,19).replace('T',' ')}</td><td>${esc(x.tool)}</td><td>${esc(JSON.stringify(x.args))}</td><td>${esc(x.status)}${x.error?' '+esc(x.error):''}</td><td>${esc(x.source)}</td><td class=n>${x.duration_ms}</td></tr>`).join('')+'</table>':'<p class=k>No MCP calls yet.</p>'}
fetch('/api/mcp/tools').then(r=>r.json()).then(d=>{$('tools').innerHTML=d.tools?'<table><tr><th>Tool</th><th>Allowed</th><th>Description</th></tr>'+d.tools.map(x=>`<tr><td>${esc(x.name)}</td><td>${x.allowed?'yes (read-only)':'no'}</td><td>${esc(x.description)}</td></tr>`).join('')+'</table>':'<p class=k>'+esc(d.error)+'</p>'});
draw();
</script>"""


LIVE_PAGE = """
<div class="card"><div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap"><span id="pill" class="pill disarmed">…</span>
<span id="why" class="k"></span></div></div>
<div class="grid" id="tiles"></div>
<div class="card" style="margin-top:14px"><b>Controls</b> <span class="k">(admin only; every action is logged in /var/lib/wallex-arena/control.jsonl)</span>
<p><button class="danger" onclick="disarm()">DISARM now (create kill file)</button></p>
<p>Arm requires Liara env <code>WALLEX_ENGINE=on</code>, <code>WALLEX_LIVE_ENABLED=1</code> and <code>WALLEX_API_KEY</code>, then the admin password and the phrase
<code>ARM LIVE TRADING</code>:<br><input id="pw" type="password" placeholder="admin password"> <input id="cf" placeholder="ARM LIVE TRADING" size="18">
<button onclick="arm()">Arm</button> <button onclick="balances()">Read balances (read-only)</button>
<button onclick="post('/api/ai-selftest')">AI self-test (1 real LLM call via LLM_PROVIDER, no trading)</button></p><pre id="msg"></pre></div>
<div class="card"><b>Accounts</b><div class="wrap" id="acc"></div></div>
<div class="card"><b>Recent decisions with reasons</b><div class="wrap" id="dec"></div></div>
<div class="card"><b>Errors</b><pre id="err"></pre></div>
<div class="card"><b>Health / connectivity self-check (from inside Liara)</b><pre id="health"></pre></div>
<script>
const $=id=>document.getElementById(id), f=(x,d=0)=>x==null?'':Number(x).toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d});
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'dashboard'},body:JSON.stringify(body||{})});$('msg').textContent=r.status+' '+await r.text();load()}
function disarm(){post('/api/disarm')} function arm(){post('/api/arm',{password:$('pw').value,confirm:$('cf').value})} function balances(){post('/api/balances')}
const dur=x=>x==null?'':(x<0?'ended '+dur(-x)+' ago':(Math.floor(x/86400)?Math.floor(x/86400)+'d ':'')+Math.floor(x%86400/3600)+'h '+Math.floor(x%3600/60)+'m');
async function load(){const d=await (await fetch('/api/live')).json(), s=d.state, c=d.challenge||{};
 $('pill').textContent=s.armed?'ARMED: REAL ORDERS POSSIBLE':'DISARMED'; $('pill').className='pill '+(s.armed?'armed':'disarmed');
 $('why').textContent=`engine ${s.engine_enabled?'enabled':'NOT RUNNING (WALLEX_ENGINE=off)'} | WALLEX_LIVE_ENABLED=${s.env_live_enabled?1:0} | API key ${s.api_key_set?'set':'missing'} | kill file ${s.kill_file_present?'present':'absent'}`;
 const st=d.status||{}, a=d.accounts[0]||{};
 const tiles=[['Engine process',Object.entries(s.engine_process||{}).map(([k,v])=>k+': '+(v.running?'running':'stopped')).join(', ')||'not started'],
  ['Status updated',st.updated_at?Math.round(d.now-st.updated_at)+' s ago':'n/a'],['Equity (TMN)',f(a.equity_tmn)],['Return',a.return_pct==null?'':f(a.return_pct,3)+'%'],
  ['Realized PnL (TMN)',f(a.realized_pnl_tmn)],['Max drawdown',a.max_drawdown_pct==null?'':f(a.max_drawdown_pct,3)+'%'],['USDT/TMN',f(st.usdt_tmn)],
  ['Challenge end'+(c.placeholder_end?' (placeholder)':''),c.end_utc||'not set'],['Time to end',dur(c.seconds_to_end)],
  ['Settlement phase',c.phase+(c.flat==null?'':'; flat: '+JSON.stringify(c.flat))],['Flatten to TMN from',c.flatten_from_utc||''],
  ['No new positions from',c.no_new_entries_from_utc||''],['Max exposure',c.max_exposure==null?'':Math.round(c.max_exposure*100)+'% of equity']];
 $('tiles').innerHTML=tiles.map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${esc(v)||'–'}</div></div>`).join('');
 $('acc').innerHTML=d.accounts.length?'<table><tr><th>Account</th><th>Equity</th><th>Return %</th><th>Realized PnL</th><th>Fills</th><th>Open positions</th><th>Open orders</th></tr>'+
  d.accounts.map(r=>`<tr><td>${esc(r.name)}</td><td class=n>${f(r.equity_tmn)}</td><td class=n>${f(r.return_pct,3)}</td><td class=n>${f(r.realized_pnl_tmn)}</td><td class=n>${r.fills}</td><td>${esc(JSON.stringify(r.open_positions))}</td><td class=n>${r.open_orders}</td></tr>`).join('')+'</table>':'<p class=k>The live engine has not run yet. Paper/shadow results are on the <a href="/">Arena results</a> page.</p>';
 $('dec').innerHTML=d.decisions.length?'<table><tr><th>UTC</th><th>Account</th><th>Type</th><th>Decision and reason</th></tr>'+d.decisions.map(x=>`<tr><td>${x.ts?new Date(x.ts*1000).toISOString().slice(5,19).replace('T',' '):''}</td><td>${esc(x.account)}</td><td>${esc(x.type)}</td><td>${esc(x.summary)}</td></tr>`).join('')+'</table>':'<p class=k>No live decisions yet.</p>';
 $('err').textContent=d.errors.length?d.errors.map(e=>JSON.stringify(e)).join('\\n'):'none'; $('health').textContent=JSON.stringify({selfcheck:d.selfcheck,status:st},null,1)}
load(); setInterval(load,15000);
</script>"""


FA_ERROR = "خطای داخلی رخ داد؛ لطفاً چند لحظه بعد دوباره تلاش کنید."
QUEUE = None   # JobQueue, created in serve()


def audit(action, ok, detail=""):
    """Control/audit trail: hash-free JSONL on the disk plus a structured `control.*` log line."""
    LIVE.mkdir(parents=True, exist_ok=True)
    with open(LIVE / "control.jsonl", "a") as f:
        f.write(json.dumps({"record": {"ts": time.time(), "type": "control", "action": action, "ok": ok, "detail": detail}}) + "\n")
    obs.log("control." + action, "warn" if action in ("arm", "disarm") else "info", detail or action, ok=ok)


def ai_selftest() -> dict:
    """One real LLM decision through the engine's own AI code path (LLM_PROVIDER + fallback); no market data, no orders."""
    from experiments.wallex_arena.ai import DEFAULT_MODELS, MAKER_SYSTEM, Budget, ClaudeTrader, MakerDecision
    tr = ClaudeTrader(DEFAULT_MODELS["boxapi"], Path("/dev/null"), Budget(LIVE / "ai-selftest-budget.sqlite", 1.0),
                      schema=MakerDecision, system_extra=MAKER_SYSTEM)   # LLM_PROVIDER picks the provider and its model
    dash = {"selftest": True, "time_utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "account": {"equity_tmn": 5000000, "positions": []},
            "markets": [{"symbol": "SUITMN", "spread_pct": 1.1, "vol24_tmn": 9e9, "global_4h_pct": 0.4},
                        {"symbol": "BTCTMN", "spread_pct": 0.1, "vol24_tmn": 9e10, "global_4h_pct": -0.2}]}
    tr._run(json.dumps(dash))
    r = tr.take() or {}
    audit("ai_selftest", bool(r.get("ok")), r.get("error", "") or f"{r.get('provider')}/{r.get('model')}")
    return {"answered_by": {"provider": r.get("provider"), "model": r.get("model")}, "llm_provider_env": os.environ.get("LLM_PROVIDER"),
            "routes": tr.routes, "clients": {k: repr(v) for k, v in tr.clients.items()}, **r}


def read_balances() -> dict:
    """Authenticated READ of the credit sub-account balances (no order endpoint is touched)."""
    from experiments.wallex_arena.live import LiveExecutor, WallexPrivateAPI
    if not os.environ.get("WALLEX_API_KEY"):
        return {"ok": False, "error": "WALLEX_API_KEY is not set"}
    api = WallexPrivateAPI(os.environ["WALLEX_API_KEY"], sub_account_id=os.environ.get("WALLEX_SUBACCOUNT_ID") or None)
    b = LiveExecutor(api, {}).balances()
    audit("read_balances", True)
    return {"ok": True, "subaccount": bool(api.sub_account_id), "balances": {k: v for k, v in b.items() if v.get("total")}}


def _job_ok(fn):
    def handler(payload):
        r = fn()
        if not r.get("ok"):
            raise RuntimeError(str(r.get("error") or "failed")[:300])   # -> retry with backoff, then dead-letter
        return r
    return handler


class SnapshotRejected(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


_PUSH_LOCK = threading.Lock()


def ingest_snapshot(body: bytes, sha_header: str | None) -> dict:
    import hashlib
    import io
    import shutil
    import tarfile
    if sha_header and not hmac.compare_digest(hashlib.sha256(body).hexdigest(), sha_header.strip().lower()):
        raise SnapshotRejected(400, "body sha256 does not match X-Snapshot-Sha256")
    with _PUSH_LOCK:
        PUSHED.mkdir(parents=True, exist_ok=True)
        tmp = PUSHED / f".incoming-{os.getpid()}-{int(time.time() * 1000)}"
        tmp.mkdir()
        try:
            total = 0
            try:
                tf = tarfile.open(fileobj=io.BytesIO(body), mode="r:gz")
                members = tf.getmembers()
            except (tarfile.TarError, OSError, EOFError) as exc:
                raise SnapshotRejected(400, f"not a gzip tar: {type(exc).__name__}")
            if len(members) > 5000:
                raise SnapshotRejected(413, "too many files")
            for m in members:
                rel = Path(m.name)
                if m.isdir():
                    continue
                if not m.isfile() or rel.is_absolute() or ".." in rel.parts or not rel.parts:
                    raise SnapshotRejected(400, f"unsafe tar member {m.name!r}")
                total += m.size
                if total > PUSH_MAX_UNPACKED:
                    raise SnapshotRejected(413, "unpacked snapshot too large")
                dest = tmp / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
            try:
                man = json.loads((tmp / "MANIFEST.json").read_text())
                files, ts = man["files"], float(man["snapshot_ts"])
            except (OSError, ValueError, KeyError, TypeError):
                raise SnapshotRejected(400, "MANIFEST.json missing or invalid")
            have = {str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file()} - {"MANIFEST.json"}
            if have != set(files):
                raise SnapshotRejected(400, f"files do not match the manifest ({len(have)} vs {len(files)})")
            for rel, meta in files.items():
                if hashlib.sha256((tmp / rel).read_bytes()).hexdigest() != meta.get("sha256"):
                    raise SnapshotRejected(400, f"checksum mismatch: {rel}")
            cur = pushed_dir()
            cur_ts = json.loads((cur / "MANIFEST.json").read_text()).get("snapshot_ts", 0) if cur else 0
            if ts <= cur_ts:
                raise SnapshotRejected(409, f"not newer than the current snapshot ({ts} <= {cur_ts})")
            man["pushed_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            (tmp / "MANIFEST.json").write_text(json.dumps(man))
            final = PUSHED / f"v{int(ts)}"
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
            link = PUSHED / ".current-new"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(final.name)
            os.replace(link, PUSHED / "current")          # atomic swap: readers see the old or the new snapshot, never a mix
            for old in sorted((p for p in PUSHED.glob("v*") if p.is_dir() and p != final), key=lambda p: p.name)[:-2]:
                shutil.rmtree(old, ignore_errors=True)    # keep the newest three versions
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    obs.log("snapshot.push", "info", "snapshot accepted", status=200, snapshot_ts=ts, files=len(files), bytes=len(body))
    return {"accepted": True, "snapshot_utc": man.get("snapshot_utc"), "snapshot_ts": ts, "files": len(files), "bytes": len(body)}


def snapshot_status() -> dict:
    """No bundled results snapshot is a valid build (repo-root image); a bundled but incomplete or unreadable one is not."""
    man = SNAP / "MANIFEST.json"
    if not SNAP.exists() or not any(SNAP.iterdir()):
        return {"ok": True, "bundled": False}
    try:
        files = json.loads(man.read_text())["files"]
    except (OSError, ValueError, KeyError):
        return {"ok": False, "bundled": True, "error": "MANIFEST.json missing or unreadable"}
    missing = [f for f in files if not os.access(SNAP / f, os.R_OK)]
    return {"ok": not missing, "bundled": True, "files": len(files), "missing_or_unreadable": missing[:5]}


def readiness() -> tuple[bool, dict]:
    from experiments.wallex_arena import resilience
    checks = {"snapshot": snapshot_status(), "pushed_snapshot": snapshot_info() if pushed_dir() else None}
    try:
        LIVE.mkdir(parents=True, exist_ok=True)
        probe = LIVE / ".ready-probe"
        probe.write_text("1")
        probe.unlink()
        checks["state_dir_writable"] = True
    except OSError:
        checks["state_dir_writable"] = False
    counts = QUEUE.counts() if QUEUE else {}
    backlog = counts.get("queued", 0) + counts.get("retrying", 0)
    checks["queue"] = {"ok": QUEUE is not None and backlog < 100, "backlog": backlog, "dead": counts.get("dead", 0)}
    provider = (os.environ.get("LLM_PROVIDER") or "anthropic").lower()
    llm = {"provider": provider, "anthropic_configured": bool(os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")),
           "boxapi_keys": len([k for k in (os.environ.get("BOXAPI_KEYS") or "").split(",") if k.strip()])}
    llm["ok"] = llm["anthropic_configured"] or llm["boxapi_keys"] > 0
    checks["llm"] = llm
    checks["circuits"] = resilience.breaker_states()
    checks["engine"] = armed_state()
    ok = checks["snapshot"]["ok"] and checks["state_dir_writable"] and checks["queue"]["ok"]
    return ok, checks


class Handler(BaseHTTPRequestHandler):
    server_version = "meowmeow-wallex"
    sys_version = ""

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8", extra=None):
        self._status, self._bytes = code, len(body)
        self.send_response(code)
        self.send_header("X-Request-ID", obs.context()["request_id"] or "")
        if obs.traceparent():
            self.send_header("traceparent", obs.traceparent())
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json")

    def _authorized(self) -> bool:
        pw = os.environ.get("WALLEX_ADMIN_PASSWORD")
        if not pw:
            return False
        expected = "Basic " + base64.b64encode(f"{USER}:{pw}".encode()).decode()
        return hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected.encode())

    def _gate(self) -> bool:
        if self._authorized():
            return True
        obs.log("auth.fail", "warn", "dashboard login required", route=urlparse(self.path).path, method=self.command,
                client_ip_hash=obs.short_hash(self.client_address[0]), had_credentials=bool(self.headers.get("Authorization")))
        self._send(401, b"authentication required", "text/plain", {"WWW-Authenticate": 'Basic realm="MeowMeow Wallex", charset="UTF-8"'})
        return False

    def _file(self, base: Path, rel: str):
        p = (base / rel).resolve()
        if not str(p).startswith(str(base.resolve()) + os.sep) or not p.is_file():
            return self._send(404, b"not found", "text/plain")
        self._send(200, p.read_bytes(), CT.get(p.suffix, "application/octet-stream"))

    # ---- every request: request id / trace context, global error handler, one http.request line
    def _dispatch(self, fn):
        obs.set_context(self.headers.get("X-Request-ID"), self.headers.get("traceparent"),
                        session=self.headers.get("Authorization", "")[-12:] or None)
        self._status, self._bytes = None, 0
        t = time.perf_counter()
        try:
            fn()
        except Exception as exc:   # never a stack trace to the client
            obs.log_error("http.error", exc, route=urlparse(self.path).path, method=self.command)
            if self._status is None:
                self._json(500, {"error": "internal_error", "request_id": obs.context()["request_id"], "message": FA_ERROR})
        finally:
            route = urlparse(self.path).path
            obs.log("http.request", "debug" if route in ("/healthz", "/readyz") and (self._status or 500) < 400 else
                    "warn" if (self._status or 500) >= 500 else "info", f"{self.command} {route}", route=route, method=self.command,
                    status=self._status, duration_ms=int((time.perf_counter() - t) * 1000), bytes=self._bytes,
                    client_ip_hash=obs.short_hash(self.client_address[0]), user_agent=(self.headers.get("User-Agent") or "")[:120])

    def do_HEAD(self):
        self._dispatch(self._get)

    def do_GET(self):
        self._dispatch(self._get)

    def do_POST(self):
        self._dispatch(self._post)

    def _get(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        if path == "/healthz":
            return self._send(200, b"ok", "text/plain")
        if path == "/readyz":
            ok, checks = readiness()
            return self._json(200 if ok else 503, {"ready": ok, **checks})
        if not self._gate():
            return
        if path == "/api/snapshot/info":
            return self._json(200, snapshot_info())
        if path in ("/", "/index.html"):
            if not sp("www/index.html").is_file():  # code-only build and nothing pushed yet
                return self._send(200, page("Wallex arena", '<div class="card"><b>No paper/shadow results snapshot yet.</b> '
                                                            'The live engine control is at <a href="/live">/live</a>.</div>'))
            raw = sp("www/index.html").read_text()
            m = manifest()
            bar = (f'<div class="mm-banner" style="background:#fff4d6;color:#4a3500;padding:9px 16px;font:13px/1.45 system-ui,sans-serif;border-bottom:1px solid #e8cf86">'
                   f'{updated_line(m)}<b>PAPER / SHADOW TRADING RESULTS</b>: snapshot of the Hetzner arena taken <b>{html.escape(m.get("snapshot_utc", "?"))}</b> '
                   f'(running since {html.escape(m.get("arena_started_utc", "?"))}); every account traded a simulated 5,000,000 TMN on live Wallex order books. No real orders. '
                   f'<div style="margin-top:4px;line-height:1.9">{NAV_INLINE}</div></div>')
            i = raw.find(">", raw.find("<body")) + 1
            h = raw.find("</head>")
            raw = raw[:h] + MOBILE_CSS + raw[h:] if h > 0 else raw
            i = raw.find(">", raw.find("<body")) + 1
            raw = raw.replace("</body>", UPDATE_JS + "</body>") if "</body>" in raw else raw + UPDATE_JS
            return self._send(200, (raw[:i] + bar + raw[i:]).encode())
        if path.startswith("/data/"):
            return self._file(sp("www/data"), path[6:])   # the arena page re-fetches these every 30 s
        if path == "/live":
            return self._send(200, page("Live engine", LIVE_PAGE))
        if path == "/api/live":
            return self._json(200, api_live())
        if path == "/market":
            return self._send(200, page("Market (MCP)", MARKET_PAGE))
        if path == "/api/market/candles":
            sym, res, hrs = (q.get("symbol") or ["BTCTMN"])[0].upper()[:20], (q.get("resolution") or ["60"])[0][:4], float((q.get("hours") or ["72"])[0])
            try:
                return self._json(200, cached(("c", sym, res, hrs), 60, lambda: mcp_client().candles(sym, res, hrs)))
            except Exception as exc:
                return self._json(502, {"error": f"{type(exc).__name__}: {str(exc)[:160]}"})
        if path == "/api/market/book":
            sym = (q.get("symbol") or ["BTCTMN"])[0].upper()[:20]
            try:
                return self._json(200, cached(("b", sym), 10, lambda: mcp_client().order_book(sym, 10)))
            except Exception as exc:
                return self._json(502, {"error": f"{type(exc).__name__}: {str(exc)[:160]}"})
        if path == "/api/mcp/calls":
            return self._json(200, mcp_calls())
        if path == "/api/mcp/tools":
            try:
                return self._json(200, {"server": mcp_client().url, "tools": cached(("tools",), 3600, lambda: mcp_client().list_tools())})
            except Exception as exc:
                return self._json(502, {"error": f"MCP server unavailable: {type(exc).__name__}"})
        if path == "/api/jobs":
            return self._json(200, {"counts": QUEUE.counts(), "jobs": QUEUE.list((q.get("state") or [None])[0], 100)})
        if path.startswith("/api/jobs/"):
            job = QUEUE.get(path.split("/")[3])
            return self._json(200 if job else 404, job or {"error": "not found"})
        if path == "/report":
            text = sp("report.md").read_text() if sp("report.md").is_file() else "report missing"
            return self._send(200, page("Shadow-test report", f'<div class="card"><p class="k">Generated from the arena logs at snapshot time. '
                                                              f'<a href="/files/report.md">raw markdown</a></p><pre>{html.escape(text)}</pre></div>'))
        if path == "/decisions":
            return self.decisions(q)
        if path == "/paper-agent":
            return self.paper_agent()
        if path in ("/docs", "/docs/"):
            items = sorted(p.name for p in sp("docs").glob("*.md")) if sp("docs").is_dir() else []
            links = "".join(f'<li><a href="/docs/{html.escape(n)}">{html.escape(n)}</a></li>' for n in items)
            bt = sp("research/backtest_tmn.txt").read_text() if sp("research/backtest_tmn.txt").is_file() else ""
            return self._send(200, page("Docs", f'<div class="card"><b>Docs</b><ul>{links}</ul></div><div class="card"><b>7-day backtest (lag rule, TMN markets)</b><pre>{html.escape(bt)}</pre></div>'
                                                f'<div class="card"><b>Snapshot manifest</b> <a href="/files/MANIFEST.json">MANIFEST.json</a> (sha256 of every file)</div>'))
        if path.startswith("/docs/"):
            p = sp("docs") / Path(path[6:]).name
            if p.is_file():
                return self._send(200, page(p.name, f'<div class="card"><pre>{html.escape(p.read_text())}</pre></div>'))
            return self._send(404, b"not found", "text/plain")
        if path.startswith("/files/"):
            if path.startswith("/files/bundled/"):
                return self._file(SNAP, path[15:])                 # the full hash-chained logs shipped in the image
            rel, cur = path[7:], pushed_dir()
            inside = cur is not None and str((cur / rel).resolve()).startswith(str(cur.resolve()) + os.sep)
            return self._file(cur if inside and (cur / rel).is_file() else SNAP, rel)   # _file re-checks containment
        self._send(404, b"not found", "text/plain")

    def decisions(self, q):
        src = (q.get("src") or ["snapshot"])[0]
        root = LIVE if src == "live" else sp("arena")
        accounts = sorted(p.name for p in (root / "accounts").iterdir() if p.is_dir()) if (root / "accounts").is_dir() else []
        acct = (q.get("account") or [""])[0]
        links = " ".join(f'<a href="/decisions?src={src}&account={html.escape(a)}">{html.escape(a)}</a>' for a in accounts)
        body = (f'<div class="card"><b>Decision logs ({"live engine" if src == "live" else "paper/shadow snapshot"})</b> '
                f'<a href="/decisions?src={"snapshot" if src == "live" else "live"}">switch to {"snapshot" if src == "live" else "live"}</a>'
                f'<p>{links or "<span class=k>no accounts</span>"}</p><p class="k">Each record is SHA-256 hash-chained to the previous one (verified in the report). '
                f'The AI records include the model, its market view and the reason for every quote or trade.</p></div>')
        if acct in accounts:
            recs = tail_jsonl(root / "accounts" / acct / "decisions.jsonl", 300)
            if src != "live" and root != SNAP / "arena":
                raw = (f'(recent tail pushed from Hetzner, AI input dashboards removed; full hash-chained log as bundled: '
                       f'<a href="/files/bundled/arena/accounts/{html.escape(acct)}/decisions.jsonl">download</a>)')
            else:
                raw = f'<a href="/files/arena/accounts/{html.escape(acct)}/decisions.jsonl">download full hash-chained log</a>' if src != "live" else ""
            body += f'<div class="card"><b>{html.escape(acct)}</b>: last {len(recs)} records {raw}{decisions_table(recs)}</div>'
        self._send(200, page("Decision logs", body))

    def paper_agent(self):
        p = sp("paper-agent")
        try:
            s = json.loads((p / "status.json").read_text())
        except (OSError, ValueError):
            s = {}
        port = s.get("portfolio") or {}
        dec = sorted((p / "decisions").rglob("*.json"))[-40:] if (p / "decisions").is_dir() else []
        items = []
        for f in reversed(dec):
            try:
                d = json.loads(f.read_text())
            except ValueError:
                continue
            items.append(f"<tr><td>{html.escape(f.stem)}</td><td><pre>{html.escape(json.dumps(d, ensure_ascii=False)[:1200])}</pre></td></tr>")
        body = (f'<div class="card"><b>Paper agent</b> (separate paper-only research agent; mode {html.escape(str(s.get("mode", "?")))}, '
                f'live execution enabled: {html.escape(str(s.get("live_execution_enabled")))}, observed {html.escape(str(s.get("observed_at", "?")))})'
                f'<pre>{html.escape(json.dumps(port, ensure_ascii=False, indent=1)[:6000])}</pre></div>'
                f'<div class="card"><b>Latest decisions</b><div class="wrap"><table>{"".join(items) or "<tr><td>none</td></tr>"}</table></div></div>')
        self._send(200, page("Paper agent", body))

    def _snapshot_push(self):
        """Ingest a newer paper/shadow results snapshot pushed from the arena host (bearer token, size-capped,
        checksum-verified, versioned by the source timestamp, atomic swap of /data/snapshots/current)."""
        token = os.environ.get("SNAPSHOT_PUSH_TOKEN")
        if not token:
            return self._json(503, {"error": "snapshot push disabled (SNAPSHOT_PUSH_TOKEN not set)"})
        if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
            obs.log("auth.fail", "warn", "snapshot push with a bad token", route="/api/snapshot/push",
                    client_ip_hash=obs.short_hash(self.client_address[0]))
            return self._json(401, {"error": "bad token"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > PUSH_MAX_BYTES:
            return self._json(413, {"error": f"body must be 1..{PUSH_MAX_BYTES} bytes"})
        body = self.rfile.read(n)
        try:
            result = ingest_snapshot(body, self.headers.get("X-Snapshot-Sha256"))
        except SnapshotRejected as exc:
            obs.log("snapshot.push", "warn", str(exc), status=exc.code)
            return self._json(exc.code, {"error": str(exc)})
        audit("snapshot_push", True, f"{result['snapshot_utc']} ({result['files']} files)")
        return self._json(200, result)

    def _post(self):
        path, q = urlparse(self.path).path, parse_qs(urlparse(self.path).query)
        if path == "/api/snapshot/push":
            return self._snapshot_push()
        if not self._gate():
            return
        if self.headers.get("X-Requested-With") != "dashboard":
            return self._json(403, {"error": "missing X-Requested-With header"})
        n = min(int(self.headers.get("Content-Length") or 0), 10_000)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        if path == "/api/disarm":
            KILL.parent.mkdir(parents=True, exist_ok=True)
            KILL.touch()
            audit("disarm", True)
            return self._json(200, {"ok": True, "state": armed_state()})
        if path == "/api/arm":
            st = armed_state()
            problems = [m for c, m in ((hmac.compare_digest(str(body.get("password", "")).encode(), os.environ.get("WALLEX_ADMIN_PASSWORD", "").encode()), "wrong admin password"),
                                       (body.get("confirm") == CONFIRM, f"type the phrase {CONFIRM!r}"),
                                       (st["engine_enabled"], "WALLEX_ENGINE is not on"), (st["env_live_enabled"], "WALLEX_LIVE_ENABLED is not 1"),
                                       (st["api_key_set"], "WALLEX_API_KEY is not set")) if not c]
            if problems:
                audit("arm", False, "; ".join(problems))
                return self._json(409, {"ok": False, "refused": problems})
            KILL.unlink(missing_ok=True)
            audit("arm", True)
            return self._json(200, {"ok": True, "state": armed_state()})
        jobs = {"/api/ai-selftest": ("ai.selftest", ai_selftest), "/api/balances": ("wallex.balances_read", read_balances)}
        if path in jobs:
            kind, fn = jobs[path]
            if (q.get("async") or ["0"])[0] == "1":      # durable background job with retries and dead-letter
                return self._json(202, {"job_id": QUEUE.enqueue(kind, {}, max_attempts=3), "poll": "/api/jobs/<job_id>"})
            try:
                r = fn()
            except Exception as exc:
                obs.log_error("job.inline_error", exc, kind=kind)
                audit(kind.split(".")[-1], False, type(exc).__name__)
                return self._json(502, {"ok": False, "error": type(exc).__name__, "message": str(exc)[:200]})
            return self._json(200 if r.get("ok") else 502 if path == "/api/ai-selftest" else 409, r)
        if path.startswith("/api/jobs/") and path.endswith(("/replay", "/discard")):
            job_id, action = path.split("/")[3], path.split("/")[4]
            ok = QUEUE.replay(job_id) if action == "replay" else QUEUE.discard(job_id)
            audit("job_" + action, ok, job_id)
            return self._json(200 if ok else 409, {"ok": ok, "job": QUEUE.get(job_id)})
        self._json(404, {"error": "not found"})


def make_server(port: int, extra_jobs: dict | None = None) -> ThreadingHTTPServer:
    """Build the server and start the durable job worker (jobs.sqlite on the state disk)."""
    global QUEUE
    from experiments.wallex_arena.jobqueue import JobQueue
    QUEUE = JobQueue(LIVE / "jobs.sqlite")
    QUEUE.register("ai.selftest", _job_ok(ai_selftest))
    QUEUE.register("wallex.balances_read", _job_ok(read_balances))
    for kind, fn in (extra_jobs or {}).items():
        QUEUE.register(kind, fn)
    QUEUE.start()
    obs.log("app.start", "info", f"web listening on {port}", port=port, admin_password_set=bool(os.environ.get("WALLEX_ADMIN_PASSWORD")))
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def serve(port: int):
    make_server(port).serve_forever()
