"""Assemble a MarketSnapshot per symbol. Every failed request becomes an entry in `errors`;
no missing value is ever replaced by a guess."""
from __future__ import annotations

import asyncio
import logging

from config import AppConfig
from src.exchanges.base import ExchangeConnector
from src.liquidation.providers import LiquidationProvider
from src.models import MarketSnapshot, RatioKind
from src.utils import TIMEFRAMES, log_event, utcnow

log = logging.getLogger("collector")


class MarketDataCollector:
    def __init__(self, connector: ExchangeConnector, provider: LiquidationProvider, cfg: AppConfig, clock=utcnow, feed=None):
        self.x, self.provider, self.cfg, self.clock, self.feed = connector, provider, cfg, clock, feed

    async def snapshot(self, symbol: str) -> MarketSnapshot:
        snap = MarketSnapshot(symbol=symbol, now=self.clock())
        kind = RatioKind(self.cfg.ratio.kind)
        jobs = {f"candles_{tf}": self.x.klines(symbol, tf, self.cfg.strategy.candle_limit) for tf in TIMEFRAMES}
        jobs.update({
            "ticker": self.x.ticker(symbol),
            "mark_funding": self.x.mark_and_funding(symbol),
            "funding_history": self.x.funding_history(symbol, 50),
            "oi_history": self.x.open_interest_history(symbol, "5m", 100),
            f"ratio_{kind.value}": self.x.ratio_history(kind, symbol, self.cfg.ratio.period, 100),
            "orderbook": self.x.order_book(symbol, 100),
            "rules": self.x.symbol_rules(symbol),
            "liquidation": self.provider.fetch_clusters(symbol),
        })
        results = await asyncio.gather(*jobs.values(), return_exceptions=True)
        got = dict(zip(jobs.keys(), results))
        for name, res in got.items():
            if isinstance(res, BaseException):
                snap.errors[name] = f"{type(res).__name__}: {str(res)[:160]}"
        ok = lambda k: not isinstance(got[k], BaseException)
        for tf in TIMEFRAMES:
            if ok(f"candles_{tf}") and len(got[f"candles_{tf}"]):
                snap.candles[tf] = got[f"candles_{tf}"]
        if ok("ticker"):
            snap.price, snap.volume_24h_usd, t = got["ticker"]
            snap.timestamps["price"] = t.to_pydatetime()
        if ok("mark_funding"):
            snap.mark_price, snap.funding = got["mark_funding"]
        if ok("funding_history"):
            snap.funding_history = got["funding_history"]
        if ok("oi_history") and len(got["oi_history"]):
            snap.oi_history = got["oi_history"]
        rk = f"ratio_{kind.value}"
        if ok(rk) and len(got[rk]):
            snap.ratio_history[kind] = got[rk]
        if ok("orderbook"):
            snap.orderbook = got["orderbook"]
        if ok("rules"):
            snap.rules = got["rules"]
        if ok("liquidation"):
            snap.liquidation_clusters = got["liquidation"] or None
            snap.liquidation_source = getattr(self.provider, "name", "provider")
            if not got["liquidation"]:
                snap.errors["liquidation"] = "provider returned no clusters"
        if self.feed is not None:
            snap.liquidation_events = self.feed.recent(symbol)          # ACTUAL events, kept separate from estimated clusters
        if snap.errors:
            log_event(log, logging.WARNING, "snapshot_missing_data", symbol=symbol, errors=snap.errors)
        return snap
