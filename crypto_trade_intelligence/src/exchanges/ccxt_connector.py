"""Generic ccxt connector for other USDT-perpetual exchanges (market data only).

Long/short ratios are NOT available through ccxt's unified API on most exchanges, so ratio history
raises NotSupported. Because of the mandatory SHORT filter, SHORT signals are then impossible on such
exchanges unless you supply the ratio from another licensed source. Nothing is inferred or faked.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd

from src.exchanges.base import ExchangeConnector, NotSupported
from src.models import OrderBook, RatioKind, SymbolRules, TimedValue
from src.utils import utcnow


class CcxtConnector(ExchangeConnector):
    def __init__(self, exchange_id: str, exchange=None):
        if exchange is None:
            import ccxt.async_support as ccxt   # lazy import
            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.ex, self.name = exchange, f"ccxt:{exchange_id}"

    @staticmethod
    def _sym(symbol: str) -> str:
        return f"{symbol[:-4]}/USDT:USDT" if symbol.endswith("USDT") else symbol

    async def klines(self, symbol, interval, limit=500, start_ms=None, end_ms=None):
        rows = await self.ex.fetch_ohlcv(self._sym(symbol), interval, since=start_ms, limit=limit)
        step = pd.Timedelta(self.ex.parse_timeframe(interval), unit="s")
        df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
        df["close_time"] = df.index + step - pd.Timedelta(milliseconds=1)
        return df.drop(columns="t")[lambda d: d["close_time"] <= pd.Timestamp(utcnow())]

    async def ticker(self, symbol):
        t = await self.ex.fetch_ticker(self._sym(symbol))
        return float(t["last"]), float(t.get("quoteVolume") or 0), pd.Timestamp(t["timestamp"], unit="ms", tz="UTC")

    async def mark_and_funding(self, symbol):
        f = await self.ex.fetch_funding_rate(self._sym(symbol))
        ts = pd.Timestamp(f["timestamp"], unit="ms", tz="UTC").to_pydatetime() if f.get("timestamp") else None
        return f.get("markPrice"), TimedValue(f.get("fundingRate"), self.name, ts)

    async def funding_history(self, symbol, limit=100):
        rows = await self.ex.fetch_funding_rate_history(self._sym(symbol), limit=limit)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.rename(columns={"fundingRate": "rate"})[["rate"]]

    async def open_interest_history(self, symbol, period="5m", limit=100):
        if not self.ex.has.get("fetchOpenInterestHistory"):
            raise NotSupported("exchange has no open-interest history via ccxt")
        rows = await self.ex.fetch_open_interest_history(self._sym(symbol), period, limit=limit)
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["oi"] = df["openInterestAmount"].astype(float)
        return df[["oi"]]

    async def ratio_history(self, kind, symbol, period="5m", limit=100):
        raise NotSupported("long/short ratio is not exposed by this exchange connector")

    async def order_book(self, symbol, limit=100):
        ob = await self.ex.fetch_order_book(self._sym(symbol), limit)
        return OrderBook([tuple(x[:2]) for x in ob["bids"]], [tuple(x[:2]) for x in ob["asks"]], utcnow(), self.name)

    async def symbol_rules(self, symbol):
        await self.ex.load_markets()
        m = self.ex.market(self._sym(symbol))
        prec, lim = m["precision"], m["limits"]
        tick = Decimal(str(prec["price"])) if prec["price"] < 1 else Decimal(1) / (Decimal(10) ** int(prec["price"]))
        step = Decimal(str(prec["amount"])) if prec["amount"] < 1 else Decimal(1) / (Decimal(10) ** int(prec["amount"]))
        return SymbolRules(symbol, tick, step, Decimal(str(lim["amount"]["min"] or step)),
                           Decimal(str(lim["cost"]["min"] or 5)), None, self.name)

    async def universe(self):
        mk = await self.ex.load_markets()
        out = []
        for m in mk.values():
            if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active", True):
                ob = m.get("info", {}).get("onboardDate")
                out.append({"symbol": f"{m['base']}USDT",
                            "onboard": pd.Timestamp(int(ob), unit="ms", tz="UTC").to_pydatetime() if ob else None})
        return out

    async def all_tickers(self):
        t = await self.ex.fetch_tickers()
        rows = {k.split("/")[0] + "USDT": {"last": v.get("last"), "quote_volume": v.get("quoteVolume") or 0.0,
                                           "change_pct": v.get("percentage") or 0.0}
                for k, v in t.items() if k.endswith(":USDT")}
        return pd.DataFrame.from_dict(rows, orient="index")
