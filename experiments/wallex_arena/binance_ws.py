"""Real-time global prices: Binance combined mini-ticker stream (1 update per second per pair).

Updates MarketFeed.binance[pair] and MarketFeed.global_hist[pair] from a background thread, so fair values and the
fast drop guard react within a second instead of waiting for the 8-second REST poll. Public data only.
"""
from __future__ import annotations

import json
import threading
import time

from websockets.sync.client import connect

URL = "wss://stream.binance.com:9443/stream?streams="


class BinanceStream:
    def __init__(self, feed):
        self.feed = feed
        self.pairs: tuple[str, ...] = ()
        self.wanted: tuple[str, ...] = ()
        self.messages = 0
        self.reconnects = 0
        self.last_message = 0.0
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="binance-stream", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def set_pairs(self, pairs):
        self.wanted = tuple(sorted(set(pairs)))

    def health(self) -> dict:
        return {"pairs": len(self.pairs), "messages": self.messages, "reconnects": self.reconnects,
                "last_message_age": round(time.time() - self.last_message, 2) if self.last_message else None,
                "recent_errors": self.errors[-3:]}

    def _run(self):
        delay = 1.0
        while not self._stop.is_set():
            pairs = self.wanted
            if not pairs:
                time.sleep(1)
                continue
            try:
                with connect(URL + "/".join(p.lower() + "@miniTicker" for p in pairs), open_timeout=10, close_timeout=2) as ws:
                    self.pairs, delay = pairs, 1.0
                    while not self._stop.is_set() and self.wanted == pairs:
                        try:
                            raw = ws.recv(timeout=30)
                        except TimeoutError:
                            break  # silent connection: reconnect
                        data = json.loads(raw).get("data") or {}
                        pair, close = data.get("s"), data.get("c")
                        if pair and close:
                            self.feed.on_global_price(pair, float(close), time.time())
                            self.messages += 1
                            self.last_message = time.time()
            except Exception as exc:  # network errors: back off and reconnect; REST polling keeps prices flowing
                self.errors.append(f"{time.strftime('%H:%M:%S')} {type(exc).__name__}")
                self.errors = self.errors[-20:]
                time.sleep(delay)
                delay = min(30.0, delay * 2)
            if not self._stop.is_set():
                self.reconnects += 1
