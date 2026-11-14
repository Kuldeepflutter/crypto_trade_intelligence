"""Live WebSocket feed (mark price, depth, public liquidation events) with auto-reconnect.

Public Binance USDⓈ-M streams only. Liquidation EVENTS are actual forced orders and are kept
separate from estimated heatmap clusters.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import deque
from datetime import datetime, timezone

from src.models import LiquidationEvent
from src.utils import log_event

log = logging.getLogger("ws")


def parse_force_order(msg: dict) -> LiquidationEvent | None:
    """Binance forceOrder: o.S == 'SELL' means a LONG position was liquidated, 'BUY' a SHORT."""
    o = msg.get("o") if msg.get("e") == "forceOrder" else None
    if not o:
        return None
    try:
        price = float(o.get("ap") or o["p"])
        qty = float(o["z"] if float(o.get("z", 0)) > 0 else o["q"])
        side = "long" if o["S"] == "SELL" else "short"
        return LiquidationEvent(o["s"], side, price, qty, datetime.fromtimestamp(int(o["T"]) / 1000, timezone.utc), "binance_usdm:forceOrder")
    except (KeyError, ValueError, TypeError):
        return None


class LiveFeed:
    def __init__(self, ws_url: str, symbols: list[str] | None = None, max_events: int = 5000):
        """symbols=None -> every symbol (single all-market streams; used for all-coins scanning)."""
        self.url, self.symbols = ws_url.rstrip("/"), ([s.lower() for s in symbols] if symbols else None)
        self.mark: dict[str, tuple[float, datetime]] = {}
        self.liquidations: deque[LiquidationEvent] = deque(maxlen=max_events)
        self._stop = False

    def _stream_url(self) -> str:
        streams = ["!markPrice@arr@1s"] if self.symbols is None else [f"{s}@markPrice@1s" for s in self.symbols]
        streams.append("!forceOrder@arr")
        return f"{self.url}/stream?streams={'/'.join(streams)}"

    def handle(self, raw: str) -> None:
        data = json.loads(raw)
        d = data.get("data", data) if isinstance(data, dict) else data
        for item in (d if isinstance(d, list) else [d]):
            if item.get("e") == "markPriceUpdate":
                self.mark[item["s"]] = (float(item["p"]), datetime.fromtimestamp(int(item["E"]) / 1000, timezone.utc))
            elif item.get("e") == "forceOrder":
                ev = parse_force_order(item)
                if ev and (self.symbols is None or ev.symbol.lower() in self.symbols):
                    self.liquidations.append(ev)

    def recent(self, symbol: str, minutes: int = 60) -> list[LiquidationEvent]:
        """ACTUAL liquidation events seen on the public stream (a sample: the stream sends at most one per second per symbol)."""
        cut = datetime.now(timezone.utc).timestamp() - minutes * 60
        return [e for e in self.liquidations if e.symbol == symbol and e.timestamp.timestamp() >= cut]

    async def run(self) -> None:
        import websockets                                  # lazy import
        attempt = 0
        while not self._stop:
            try:
                async with websockets.connect(self._stream_url(), ping_interval=20, max_queue=1024) as ws:
                    attempt = 0
                    async for raw in ws:
                        self.handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                attempt += 1
                wait = min(60.0, 1.0 * 2 ** min(attempt, 6)) + random.uniform(0, 1)
                log_event(log, logging.WARNING, "ws_reconnect", error=type(e).__name__, wait_s=round(wait, 1))
                await asyncio.sleep(wait)

    def stop(self) -> None:
        self._stop = True
