"""Binance USDⓈ-M Futures public market-data connector (no API key needed for these endpoints)."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from src.exchanges.base import ExchangeConnector
from src.exchanges.http import ExchangeError, HttpClient
from src.models import OrderBook, RatioKind, SymbolRules, TimedValue
from src.utils import utcnow

RATIO_PATHS = {
    RatioKind.GLOBAL_ACCOUNT: "/futures/data/globalLongShortAccountRatio",
    RatioKind.TOP_TRADER_ACCOUNT: "/futures/data/topLongShortAccountRatio",
    RatioKind.TOP_TRADER_POSITION: "/futures/data/topLongShortPositionRatio",
}


def _ts(ms) -> pd.Timestamp:
    return pd.to_datetime(int(ms), unit="ms", utc=True)


def kline_weight(limit: int) -> int:
    """Binance request weight for /fapi/v1/klines by `limit`."""
    return 1 if limit < 100 else (2 if limit < 500 else (5 if limit <= 1000 else 10))


def depth_weight(limit: int) -> int:
    return 2 if limit <= 50 else (5 if limit <= 100 else (10 if limit <= 500 else 20))


def parse_universe(info: dict) -> list[dict]:
    """TRADING USDT-margined PERPETUAL contracts only (no delivery futures, no settled/pre-trading symbols)."""
    out = []
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        ob = s.get("onboardDate")
        out.append({"symbol": s["symbol"], "onboard": _ts(ob).to_pydatetime() if ob else None})
    return out


def parse_tickers(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["last", "quote_volume", "change_pct"])
    out = pd.DataFrame({"last": df["lastPrice"].astype(float), "quote_volume": df["quoteVolume"].astype(float),
                        "change_pct": df["priceChangePercent"].astype(float)})
    out.index = df["symbol"]
    return out


def parse_klines(rows: list, now: datetime | None = None) -> pd.DataFrame:
    """Binance kline rows -> candle frame. The still-forming candle is dropped (closed candles only)."""
    now = now or utcnow()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "close_time"])
    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    df.index = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df.index.name = "open_time"
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    out = df[["open", "high", "low", "close", "volume", "close_time"]].copy()
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = out[c].astype(float)
    return out[out["close_time"] <= pd.Timestamp(now)]


def parse_symbol_rules(info: dict, symbol: str) -> SymbolRules:
    sym = next((s for s in info.get("symbols", []) if s.get("symbol") == symbol), None)
    if sym is None:
        raise ExchangeError(f"{symbol} not found in exchangeInfo")
    f = {x["filterType"]: x for x in sym.get("filters", [])}
    try:
        lot = f["LOT_SIZE"]
        mkt = f.get("MARKET_LOT_SIZE", lot)
        return SymbolRules(symbol, Decimal(f["PRICE_FILTER"]["tickSize"]), Decimal(lot["stepSize"]), Decimal(lot["minQty"]),
                           Decimal(f.get("MIN_NOTIONAL", {}).get("notional", "5")),
                           Decimal(mkt["maxQty"]) if mkt.get("maxQty") else None, "binance_usdm:exchangeInfo")
    except KeyError as e:
        raise ExchangeError(f"exchangeInfo for {symbol} missing filter {e}") from e


class BinanceUSDM(ExchangeConnector):
    name = "binance_usdm"

    def __init__(self, http: HttpClient):
        self.http = http

    async def klines(self, symbol, interval, limit=500, start_ms=None, end_ms=None):
        p = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms:
            p["startTime"] = start_ms
        if end_ms:
            p["endTime"] = end_ms
        ttl = {"5m": 20, "15m": 45, "1h": 120, "4h": 300}.get(interval, 30)
        return parse_klines(await self.http.get_json("/fapi/v1/klines", p, ttl=ttl, weight=kline_weight(limit)))

    async def ticker(self, symbol):
        d = await self.http.get_json("/fapi/v1/ticker/24hr", {"symbol": symbol}, ttl=5)
        return float(d["lastPrice"]), float(d["quoteVolume"]), _ts(d["closeTime"])

    async def all_tickers(self):
        return parse_tickers(await self.http.get_json("/fapi/v1/ticker/24hr", ttl=30, weight=40))

    async def universe(self):
        return parse_universe(await self.http.get_json("/fapi/v1/exchangeInfo", ttl=3600))

    async def mark_and_funding(self, symbol):
        d = await self.http.get_json("/fapi/v1/premiumIndex", {"symbol": symbol}, ttl=5)
        ts = _ts(d["time"]).to_pydatetime()
        fr = d.get("lastFundingRate")
        return float(d["markPrice"]), TimedValue(float(fr) if fr not in (None, "") else None, "binance_usdm:premiumIndex", ts)

    async def funding_history(self, symbol, limit=100, start_ms=None):
        p = {"symbol": symbol, "limit": limit, **({"startTime": start_ms} if start_ms else {})}
        rows = await self.http.get_json("/fapi/v1/fundingRate", p, ttl=300)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df.index = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
        df["rate"] = df["fundingRate"].astype(float)
        return df[["rate"]]

    async def open_interest_history(self, symbol, period="5m", limit=100, end_ms=None):
        p = {"symbol": symbol, "period": period, "limit": limit, **({"endTime": end_ms} if end_ms else {})}
        rows = await self.http.get_json("/futures/data/openInterestHist", p, ttl=60, bucket="data")
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df.index = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["oi"] = df["sumOpenInterest"].astype(float)
        df["oi_value"] = df["sumOpenInterestValue"].astype(float)
        return df[["oi", "oi_value"]].sort_index()

    async def ratio_history(self, kind, symbol, period="5m", limit=100, end_ms=None):
        p = {"symbol": symbol, "period": period, "limit": limit, **({"endTime": end_ms} if end_ms else {})}
        rows = await self.http.get_json(RATIO_PATHS[kind], p, ttl=60, bucket="data")
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df.index = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["ratio"] = df["longShortRatio"].astype(float)
        return df[["ratio"]].sort_index()

    async def order_book(self, symbol, limit=100):
        d = await self.http.get_json("/fapi/v1/depth", {"symbol": symbol, "limit": limit}, ttl=2, weight=depth_weight(limit))
        t = d.get("T") or d.get("E")
        return OrderBook([(float(p), float(q)) for p, q in d["bids"]], [(float(p), float(q)) for p, q in d["asks"]],
                         _ts(t).to_pydatetime() if t else utcnow(), "binance_usdm:depth")

    async def symbol_rules(self, symbol):
        return parse_symbol_rules(await self.http.get_json("/fapi/v1/exchangeInfo", ttl=3600), symbol)
