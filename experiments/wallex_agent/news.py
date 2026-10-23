"""Bounded public RSS evidence with point-in-time availability and article versions.

Feeds are evidence only. Their content never grants tools or execution permissions.
Historical snapshots are limited to articles this collector had actually observed.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import html
from html.parser import HTMLParser
import re
import sqlite3
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
import xml.etree.ElementTree as ET

FEEDS = (
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", "coindesk.com"),
    ("Cointelegraph", "https://cointelegraph.com/rss", "cointelegraph.com"),
    ("Decrypt", "https://decrypt.co/feed", "decrypt.co"),
    ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", "federalreserve.gov"),
    ("ECB", "https://www.ecb.europa.eu/rss/press.html", "ecb.europa.eu"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml", "bbc.com"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", "bbc.com"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", "aljazeera.com"),
    ("IRNA Economy", "https://www.irna.ir/rss/tp/20", "irna.ir"),
)
FEED_TOPICS = {"CoinDesk": "crypto", "Cointelegraph": "crypto", "Decrypt": "crypto",
               "Federal Reserve": "macro", "ECB": "macro", "BBC Business": "macro",
               "BBC World": "politics", "Al Jazeera": "politics", "IRNA Economy": "macro"}
MAX_FEED_BYTES = 2_000_000
MAX_AGE_SECONDS = 72 * 3600


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _plain(value, limit):
    parser = _Text()
    parser.feed(html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()[:limit]


def _epoch(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, datetime):
        result = value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
    elif isinstance(value, str):
        try:
            result = parsedate_to_datetime(value).timestamp()
        except (ValueError, TypeError, OverflowError):
            result = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
                tzinfo=datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo or timezone.utc
            ).timestamp()
    else:
        raise ValueError("invalid_timestamp")
    if not 0 <= result < 100_000_000_000:
        raise ValueError("invalid_timestamp")
    return result


def _canonical_url(value, domain):
    u = urlsplit(value.strip())
    host = (u.hostname or "").lower().removeprefix("www.")
    if u.scheme not in {"http", "https"} or host not in ({domain, "bbc.co.uk"} if domain == "bbc.com" else {domain}) or u.username or u.password or u.port not in {None, 80, 443}:
        raise ValueError("unapproved_article_url")
    query = [(k, v) for k, v in parse_qsl(u.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid", "ref"}]
    return urlunsplit(("https", host, u.path.rstrip("/") or "/", urlencode(sorted(query)), ""))


class _FeedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        allowed = {x[2] for x in FEEDS} | {(urlsplit(x[1]).hostname or "").removeprefix("www.") for x in FEEDS}
        u = urlsplit(newurl)
        if u.scheme != "https" or (u.hostname or "").lower().removeprefix("www.") not in allowed or u.username or u.password:
            raise ValueError("unapproved_feed_redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url):
    if url not in {x[1] for x in FEEDS}:
        raise ValueError("unapproved_feed_url")
    req = Request(url, headers={"User-Agent": "WallexResearch/1.0", "Accept": "application/rss+xml,application/atom+xml,application/xml,text/xml"})
    with build_opener(_FeedRedirect()).open(req, timeout=15) as response:
        data = response.read(MAX_FEED_BYTES + 1)
    if len(data) > MAX_FEED_BYTES:
        raise ValueError("feed_too_large")
    return data


def _items(raw, source, domain, observed_at):
    if not isinstance(raw, bytes) or len(raw) > MAX_FEED_BYTES:
        raise ValueError("feed_too_large")
    # Reject alternate encodings, DTDs and custom entities before ElementTree can
    # process them. RSS feeds in the fixed allowlist are UTF-8.
    decoded = raw.decode("utf-8-sig")
    if "\x00" in decoded or "<!DOCTYPE" in decoded.upper() or "<!ENTITY" in decoded.upper():
        raise ValueError("unsafe_xml")
    declared = re.search(r'<\?xml[^>]*encoding=["\']([^"\']+)', decoded[:200], re.I)
    if declared and declared.group(1).lower().replace("-", "") != "utf8":
        raise ValueError("unsupported_xml_encoding")
    root = ET.fromstring(decoded)
    entries = [x for x in root.iter() if x.tag.rsplit("}", 1)[-1] in {"item", "entry"}]
    result = []
    for item in entries[:500]:
        fields = {}
        for child in item:
            key = child.tag.rsplit("}", 1)[-1]
            value = "".join(child.itertext())
            if key == "link" and child.attrib.get("href"):
                if child.attrib.get("rel", "alternate") != "alternate":
                    continue
                value = child.attrib["href"]
            fields.setdefault(key, value)
        title = _plain(fields.get("title", ""), 400)
        if not title:
            continue
        try:
            url = _canonical_url(fields.get("link", ""), domain)
        except ValueError:
            continue
        published = fields.get("pubDate") or fields.get("published") or fields.get("updated")
        try:
            published_at = _epoch(published) if published else observed_at
        except (ValueError, TypeError, OverflowError):
            published_at = observed_at
        summary = _plain(fields.get("description") or fields.get("summary") or fields.get("encoded") or fields.get("content"), 2000)
        content = "\n".join((title, summary, str(published_at)))
        result.append({"source": source, "url": url, "title": title, "summary": summary,
                       "published_at": published_at, "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                       "title_key": re.sub(r"[^\w]+", " ", title.casefold()).strip()})
    return result


# Full project names are safer than matching ordinary short words. Tickers with
# common-language meanings require a financial context or explicit $/pair notation.
ASSET_ALIASES = {
    "BTC": ("bitcoin", "بیت کوین", "بیت‌کوین"),
    "ETH": ("ethereum", "ether", "اتریوم"),
    "SOL": ("solana", "سولانا"), "XRP": ("ripple", "ریپل"),
    "DOGE": ("dogecoin", "دوج کوین", "دوج‌کوین"), "ADA": ("cardano",),
    "LTC": ("litecoin",), "BCH": ("bitcoin cash",), "BNB": ("binance coin", "bnb chain"),
    "TRX": ("tron",), "DOT": ("polkadot",), "AVAX": ("avalanche",),
    "LINK": ("chainlink",), "SHIB": ("shiba inu",), "PEPE": ("pepe coin",),
    "SUI": ("sui network",), "ETC": ("ethereum classic",), "ARB": ("arbitrum",),
    "XLM": ("stellar",), "UNI": ("uniswap",), "AAVE": ("aave",),
    "LUNC": ("terra classic",), "NEAR": ("near protocol",),
    "TON": ("toncoin", "the open network"), "ATOM": ("cosmos",),
    "HBAR": ("hedera",), "ALGO": ("algorand",), "INJ": ("injective",),
    "FIL": ("filecoin",), "ICP": ("internet computer",), "OP": ("optimism network",),
    "MATIC": ("polygon",), "POL": ("polygon",), "APT": ("aptos",),
    "FTM": ("fantom",), "RUNE": ("thorchain",), "TIA": ("celestia",),
    "WIF": ("dogwifhat",), "BONK": ("bonk coin",), "FLOKI": ("floki inu",),
    "GRT": ("the graph",), "BTT": ("bittorrent",), "JASMY": ("jasmycoin",),
    "WBTC": ("wrapped bitcoin",), "WETH": ("wrapped ether",),
    "LUNA": ("terra luna",), "USDT": ("tether",), "USDC": ("usd coin",),
}
AMBIGUOUS_TICKERS = {"DOT", "LINK", "NEAR", "UNI", "SOL", "ETC", "ADA", "ATOM", "OP", "FIL", "TON", "POL", "INJ", "RUNE", "TIA", "WIF", "APT"}
AMBIGUOUS_ALIASES = {"avalanche", "stellar", "cosmos", "ether", "polygon", "the graph"}
FINANCIAL_CONTEXT = re.compile(r"\b(price|trading|market|futures|token|tokens|coin|coins|crypto|cryptocurrency|protocol|blockchain|rally|rallies|rallied|surge|surges|gains|whale|wallet|exchange|investors|staking|defi|etf)\b|ارز|رمز|قیمت", re.I)


def _base_symbol(symbol):
    value = re.sub(r"[-_/\s]", "", str(symbol).strip().upper())
    for quote in ("USDT", "TMN", "IRT", "USD"):
        if value.endswith(quote) and len(value) > len(quote):
            value = value[:-len(quote)]
            break
    return value if re.fullmatch(r"[A-Z0-9]{2,15}", value) else None


def _asset_match(title, summary, symbol, source=""):
    base = _base_symbol(symbol)
    if not base:
        return False
    text = title + " " + summary
    for alias in ASSET_ALIASES.get(base, ()):
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I):
            if alias in AMBIGUOUS_ALIASES and FEED_TOPICS.get(source) != "crypto" and not FINANCIAL_CONTEXT.search(text):
                continue
            return True
    # Dollar prefixes and exchange-style quote pairs disambiguate short tickers.
    if re.search(r"(?<!\w)\$" + re.escape(base) + r"(?!\w)|(?<!\w)" + re.escape(base) + r"[/_-]?(?:USDT|TMN|IRT|USD)(?!\w)", text, re.I):
        return True
    # Generic symbols are supported without accidentally matching word substrings.
    ticker = re.search(r"(?<!\w)" + re.escape(base) + r"(?!\w)", text)
    if not ticker:
        return False
    return base in ASSET_ALIASES and base not in AMBIGUOUS_TICKERS or bool(FINANCIAL_CONTEXT.search(text))


def _relevant(title, summary, symbol, source=""):
    text = (title + " " + summary).lower()
    macro = r"\b(crypto|cryptocurrency|cryptocurrencies|stablecoin|stablecoins|federal reserve|fed|ecb|central bank|interest rates|inflation|recession|payrolls|sec|exchange|exchanges|regulation|hack|hacked|tariff|war|sanctions|oil|treasury|bond yields|dollar|euro|sterling|yen|yuan|forex|currency|currencies|geopolitical|gdp|ceasefire)\b|تورم|نرخ بهره|بانک مرکزی|دلار|ارز|ریال|تحریم|جنگ|نفت"
    return bool(_asset_match(title, summary, symbol, source) or re.search(macro, text) or source in {"Federal Reserve", "ECB"})


class NewsStore:
    def __init__(self, db_path, fetcher=None, max_age_seconds=MAX_AGE_SECONDS):
        self.db_path = str(db_path)
        self.fetcher = fetcher or _fetch
        self.max_age_seconds = max_age_seconds
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS news_versions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, url TEXT NOT NULL,
                title TEXT NOT NULL, summary TEXT NOT NULL, title_key TEXT NOT NULL,
                content_hash TEXT NOT NULL, published_at REAL NOT NULL,
                first_seen_at REAL NOT NULL, available_at REAL NOT NULL, last_seen_at REAL NOT NULL, topic TEXT NOT NULL)""")
            if "topic" not in {r[1] for r in db.execute("PRAGMA table_info(news_versions)")} :
                db.execute("ALTER TABLE news_versions ADD COLUMN topic TEXT NOT NULL DEFAULT 'crypto'")
            db.execute("CREATE INDEX IF NOT EXISTS news_by_url ON news_versions(source,url,first_seen_at)")
            db.execute("CREATE INDEX IF NOT EXISTS news_availability ON news_versions(available_at)")

    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def refresh(self, now=None):
        observed_at = _epoch(time.time() if now is None else now)
        counts = {"fetched": 0, "inserted": 0, "unchanged": 0, "feeds": [], "observed_at": observed_at}
        def fetch_one(feed):
            try:
                raw = self.fetcher(feed[1])
                seen_at = observed_at if now is not None else _epoch(time.time())
                return _items(raw, feed[0], feed[2], seen_at), seen_at
            except Exception:
                return None, observed_at
        # Only networking/parsing is concurrent. SQLite mutations remain sequential.
        with ThreadPoolExecutor(max_workers=4) as pool:
            fetched = list(pool.map(fetch_one, FEEDS))
        for (source, url, domain), (items, seen_at) in zip(FEEDS, fetched):
            observed_at = seen_at
            try:
                if items is None:
                    raise ValueError("feed_unavailable")
                with self._db() as db:
                    db.execute("BEGIN IMMEDIATE")
                    for item in items:
                        latest = db.execute("SELECT * FROM news_versions WHERE source=? AND url=? ORDER BY first_seen_at DESC, rowid DESC LIMIT 1", (source, item["url"])).fetchone()
                        if latest and observed_at < latest["first_seen_at"]:
                            raise ValueError("clock_moved_backwards")
                        if latest and latest["content_hash"] == item["content_hash"]:
                            db.execute("UPDATE news_versions SET last_seen_at=MAX(last_seen_at,?) WHERE id=?", (observed_at, latest["id"]))
                            counts["unchanged"] += 1
                            continue
                        identity = hashlib.sha256((source + "\n" + item["url"] + "\n" + item["content_hash"] + "\n" + str(observed_at)).encode()).hexdigest()[:24]
                        db.execute("INSERT OR IGNORE INTO news_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (identity, source, item["url"], item["title"], item["summary"], item["title_key"], item["content_hash"], item["published_at"], observed_at, max(item["published_at"], observed_at), observed_at, "forex" if re.search(r"\b(dollar|euro|sterling|yen|yuan|forex|currency|currencies)\b|دلار|ارز|ریال", item["title"], re.I) else FEED_TOPICS[source]))
                        counts["inserted"] += 1
                counts["fetched"] += len(items)
                counts["feeds"].append({"source": source, "status": "ok", "items": len(items)})
            except Exception:
                # Do not leak server paths, proxy credentials, response bodies or arbitrary parser data.
                counts["feeds"].append({"source": source, "status": "unavailable", "items": 0})
        counts["status"] = "ok" if all(x["status"] == "ok" for x in counts["feeds"]) else "partial"
        counts["completed_at"] = _epoch(time.time()) if now is None else _epoch(now)
        return counts

    def snapshot(self, symbol, as_of, limit=12):
        at = _epoch(as_of)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("invalid_limit")
        # Select the latest content actually seen by as_of, even if its publication
        # timestamp makes that new version unavailable. Never fall back to superseded content.
        with self._db() as db:
            rows = db.execute("""SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY source,url ORDER BY first_seen_at DESC,rowid DESC) AS version_rank
                FROM news_versions WHERE first_seen_at <= ?)
                WHERE version_rank=1 AND available_at <= ? AND published_at >= ?
                ORDER BY available_at DESC,id""", (at, at, at-self.max_age_seconds)).fetchall()
        # Policy/rates/forex/geopolitical headlines get priority; low-relevance general
        # news is excluded before limiting. Crypto and own-asset evidence retain priority.
        def rank(row):
            text = row["title"].lower()
            score = 3 if _asset_match(row["title"], row["summary"], symbol, row["source"]) else 0
            score += 2 if re.search(r"interest rate|federal reserve|\bfed\b|\becb\b|inflation|sanction|\bwar\b|ceasefire|forex|currency|dollar|تورم|نرخ بهره|بانک مرکزی|تحریم|جنگ|دلار", text) else 0
            return (score, row["available_at"], row["id"])
        rows = sorted(rows, key=rank, reverse=True)
        records = []
        by_url, by_title = {}, {}
        for row in rows:
            if not _relevant(row["title"], row["summary"], symbol, row["source"]):
                continue
            existing = by_url.get(row["url"]) or by_title.get(row["title_key"])
            attribution = {"source": row["source"], "url": row["url"], "evidence_id": row["id"]}
            if existing is not None:
                if attribution not in existing["attribution"]:
                    existing["attribution"].append(attribution)
                by_url[row["url"]] = existing
                by_title[row["title_key"]] = existing
                continue
            evidence = {k: row[k] for k in ("id", "source", "title", "summary", "url", "published_at", "first_seen_at", "available_at", "topic")}
            evidence["attribution"] = [attribution]
            records.append(evidence)
            by_url[row["url"]] = evidence
            by_title[row["title_key"]] = evidence
        return records[:limit]
