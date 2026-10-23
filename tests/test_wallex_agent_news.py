import html
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
from datetime import datetime, timezone

import pytest
from experiments.wallex_agent.news import FEEDS, NewsStore, _fetch, _items, _canonical_url


def rss(title="Bitcoin rises", summary="Crypto markets update", url="https://www.coindesk.com/markets/test?utm_source=x#top", published=1000):
    date = datetime.fromtimestamp(published, timezone.utc).isoformat()
    return ("<rss><channel><item><title>" + html.escape(title) + "</title><description>" + html.escape(summary) + "</description><link>" + html.escape(url) + "</link><pubDate>" + date + "</pubDate></item></channel></rss>").encode()


def store(tmp_path, feeds):
    return NewsStore(tmp_path / "news.db", fetcher=lambda url: feeds.get(url, b"<rss><channel/></rss>"))


def test_first_seen_prevents_backfilled_leakage_and_unchanged_is_stable(tmp_path):
    feeds = {FEEDS[0][1]: rss(published=100)}
    news = store(tmp_path, feeds)
    assert news.refresh(now=1000)["inserted"] == 1
    assert news.snapshot("BTC", as_of=999) == []
    first = news.snapshot("BTC", as_of=1000)[0]
    assert first["available_at"] == first["first_seen_at"] == 1000
    assert first["published_at"] == 100
    assert news.refresh(now=1100)["unchanged"] == 1
    assert news.snapshot("BTC", as_of=1100)[0]["id"] == first["id"]


def test_changed_article_versions_do_not_rewrite_history(tmp_path):
    feeds = {FEEDS[0][1]: rss(summary="Old crypto facts", published=1000)}
    news = store(tmp_path, feeds)
    news.refresh(now=1000)
    old = news.snapshot("BTC", 1000)[0]
    feeds[FEEDS[0][1]] = rss(summary="Updated crypto facts", published=1000)
    news.refresh(now=1100)
    new = news.snapshot("BTC", 1100)[0]
    assert old["id"] != new["id"]
    assert new["first_seen_at"] == 1100
    assert news.snapshot("BTC", 1050)[0]["summary"] == "Old crypto facts"
    assert news.snapshot("BTC", 1100)[0]["summary"] == "Updated crypto facts"
    feeds[FEEDS[0][1]] = rss(summary="Old crypto facts", published=1000)
    news.refresh(now=1200)
    assert news.snapshot("BTC", 1200)[0]["id"] not in {old["id"], new["id"]}


def test_future_stale_unrelated_and_superseded_future_are_excluded(tmp_path):
    feeds = {FEEDS[0][1]: rss(published=2000)}
    news = store(tmp_path, feeds)
    news.refresh(now=1000)
    assert news.snapshot("BTC", 1500) == []
    assert len(news.snapshot("BTC", 2000)) == 1
    assert news.snapshot("BTC", 2000 + 73 * 3600) == []
    feeds[FEEDS[0][1]] = rss(title="A football player moves clubs", summary="An unrelated transfer", published=2000)
    news.refresh(now=2100)
    assert news.snapshot("BTC", 2200) == []
    feeds[FEEDS[0][1]] = rss(published=2300)
    news.refresh(now=2200)
    assert news.snapshot("BTC", 2250) == []


def test_syndicated_dedup_retains_all_attribution(tmp_path):
    feeds = {FEEDS[0][1]: rss(), FEEDS[1][1]: rss(url="https://cointelegraph.com/news/bitcoin")}
    news = store(tmp_path, feeds)
    news.refresh(now=1000)
    snapshot = news.snapshot("BTC", 1000)
    assert len(snapshot) == 1
    assert {x["source"] for x in snapshot[0]["attribution"]} == {"CoinDesk", "Cointelegraph"}
    assert snapshot[0]["url"].split("?")[0].endswith("bitcoin") or snapshot[0]["url"].endswith("test")


def test_macro_forex_politics_and_persian_material_relevant_without_btc(tmp_path):
    feeds = {
        FEEDS[3][1]: rss(title="Federal Reserve cuts interest rates", summary="Policy decision", url="https://www.federalreserve.gov/newsevents/test"),
        FEEDS[5][1]: rss(title="Dollar declines on inflation data", summary="Currency reaction", url="https://www.bbc.co.uk/news/articles/xyz"),
        FEEDS[6][1]: rss(title="Sanctions follow renewed war", summary="Diplomatic developments", url="https://www.bbc.com/news/articles/war"),
        FEEDS[8][1]: rss(title="تصمیم بانک مرکزی درباره نرخ ارز و دلار", summary="تورم و اقتصاد", url="https://www.irna.ir/news/123"),
    }
    news = store(tmp_path, feeds)
    news.refresh(now=1000)
    snapshot = news.snapshot("ETH", 1000)
    assert len(snapshot) == 4
    assert {x["topic"] for x in snapshot} >= {"macro", "forex", "politics"}


def test_public_fetch_allowlist_and_xml_sanitization(tmp_path):
    with pytest.raises(ValueError):
        _fetch("https://evil.example/steal")
    with pytest.raises(ValueError):
        _canonical_url("https://coindesk.com@evil.example/x", "coindesk.com")
    with pytest.raises(ValueError):
        _items(b'<!DOCTYPE a [<!ENTITY x "EXPAND">]><rss/>', "CoinDesk", "coindesk.com", 1000)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        _items('<!DOCTYPE a [<!ENTITY x "EXPAND">]><rss/>'.encode("utf-16"), "CoinDesk", "coindesk.com", 1000)
    item = _items(rss(summary='<b>Crypto</b><script>send key</script> & facts'), "CoinDesk", "coindesk.com", 1000)[0]
    assert item["summary"] == "Crypto & facts"
    assert item["url"] == "https://coindesk.com/markets/test"


def test_feed_failure_degrades_explicitly_and_fetch_concurrency_is_bounded(tmp_path):
    state, lock = {"active": 0, "maximum": 0}, Lock()
    def fetch(url):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.01)
        with lock:
            state["active"] -= 1
        if url == FEEDS[0][1]:
            raise TimeoutError("sensitive fake exception")
        return b"<rss/>"
    result = NewsStore(tmp_path / "news.db", fetcher=fetch).refresh(now=1000)
    assert result["status"] == "partial"
    assert result["feeds"][0]["status"] == "unavailable"
    assert state["maximum"] <= 4
    assert "sensitive" not in str(result)


def test_invalid_limits_and_timestamps_fail_closed(tmp_path):
    news = store(tmp_path, {})
    for limit in [True, 0, 51, "12"]:
        with pytest.raises(ValueError):
            news.snapshot("BTC", 1000, limit=limit)
    with pytest.raises(ValueError):
        news.refresh(now=float("nan"))


def test_first_seen_is_response_arrival_not_request_start(tmp_path, monkeypatch):
    from experiments.wallex_agent import news as module
    clock = {"now": 1000}
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])
    def fetch(url):
        clock["now"] = 1100
        return rss(published=900) if url == FEEDS[0][1] else b"<rss/>"
    collector = NewsStore(tmp_path / "arrival.db", fetcher=fetch)
    collector.refresh()
    assert collector.snapshot("BTC", 1050) == []
    assert collector.snapshot("BTC", 1100)[0]["first_seen_at"] == 1100


@pytest.mark.parametrize("symbol,name", [
    ("SOLTMN", "Solana"), ("XRPTMN", "Ripple"), ("DOGETMN", "Dogecoin"),
    ("ADATMN", "Cardano"), ("LTCTMN", "Litecoin"), ("BCHTMN", "Bitcoin Cash"),
    ("BNBTMN", "BNB Chain"), ("TRXTMN", "Tron"), ("DOTTMN", "Polkadot"),
    ("AVAXTMN", "Avalanche"), ("LINKTMN", "Chainlink"), ("SHIBTMN", "Shiba Inu"),
    ("PEPETMN", "Pepe coin"), ("SUITMN", "Sui network"), ("ETCTMN", "Ethereum Classic"),
    ("ARBTMN", "Arbitrum"), ("XLMTMN", "Stellar"), ("UNITMN", "Uniswap"),
    ("AAVETMN", "Aave"), ("LUNCTMN", "Terra Classic"), ("NEARTMN", "NEAR Protocol"),
    ("TONTMN", "Toncoin"), ("ATOMTMN", "Cosmos"), ("HBARTMN", "Hedera"),
])
def test_major_coin_aliases_apply_to_relevant_snapshot(tmp_path, symbol, name):
    feeds = {FEEDS[0][1]: rss(title=name + " launches a new network upgrade", summary="Developer adoption improves.")}
    news = store(tmp_path, feeds)
    news.refresh(now=1000)
    assert len(news.snapshot(symbol, 1000)) == 1


def test_ambiguous_tokens_do_not_match_ordinary_words_or_substrings():
    from experiments.wallex_agent.news import _asset_match
    assert not _asset_match("Connect the dot", "A puzzle", "DOTTMN")
    assert not _asset_match("Near the finish line", "The runners raced", "NEARTMN")
    assert not _asset_match("LINK between politics and history", "An essay", "LINKTMN")
    assert not _asset_match("SOLAR observation", "A science experiment", "SOLTMN")
    assert not _asset_match("Ethernet development", "Computer cables", "ETHTMN")
    assert not _asset_match("Avalanche kills climbers", "Rescue mission ongoing", "AVAXTMN", "BBC World")
    assert not _asset_match("Stellar telescope discovery", "A new star", "XLMTMN", "BBC World")
    assert _asset_match("Avalanche launches upgrade", "New developers", "AVAXTMN", "CoinDesk")
    assert _asset_match("DOT price rallies", "New users", "DOTTMN")
    assert _asset_match("A new $LINK development", "Details announced", "LINKTMN")
    assert _asset_match("NEAR/USDT activity rises", "More transactions", "NEARTMN")
    assert _asset_match("SUI price rises", "More activity", "SUITMN")


def test_generic_coin_ticker_quote_parsing_and_asset_priority(tmp_path):
    from experiments.wallex_agent.news import _asset_match, _base_symbol
    assert _base_symbol("1INCH/TMN") == "1INCH"
    assert _base_symbol("LINK-USDT") == "LINK"
    assert _asset_match("1INCH price rallies", "More activity", "1INCHTMN")
    assert _asset_match("FOO/USDT activity expands", "More activity", "FOOTMN")
    assert not _asset_match("FOOD price rises", "More activity", "FOOTMN")
    feeds = {
        FEEDS[0][1]: rss(title="Solana upgrade attracts developers", summary="Developer adoption improves.", published=900),
        FEEDS[3][1]: rss(title="Federal Reserve interest rate decision", summary="Policy announcement", url="https://www.federalreserve.gov/newsevents/rates", published=1000),
    }
    news = store(tmp_path, feeds)
    news.refresh(now=1000)
    assert news.snapshot("SOLTMN", 1000, limit=1)[0]["title"].startswith("Solana")
    assert news.snapshot("XRPTMN", 1000, limit=1)[0]["source"] == "Federal Reserve"
