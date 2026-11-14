"""Synthetic-market helpers for tests (NEVER used by the application itself)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import AppConfig
from src.indicators.core import resample_ohlcv
from src.liquidation.heatmap import analyze_clusters
from src.market_structure.swings import find_levels, merge_levels
from src.market_structure.trend import analyze_mtf
from src.models import LiquidationCluster, Outcome, RatioKind, RatioReading
from src.strategies.engine import StrategyEngine, StrategyInput

START = pd.Timestamp("2025-01-01", tz="UTC")
_CACHE: dict = {}


def default_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.strategy.liquidation_mode = "optional"
    cfg.strategy.require_orderbook = False
    cfg.universe.mode = "fixed"                                  # tests opt in to all-coins mode explicitly
    cfg.liquidation.provider = "none"
    return cfg


def _frame(o, h, l, c, v, start=START) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(c), freq="5min", tz="UTC")
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    df["close_time"] = df.index + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
    return df


def base_uptrend(n: int = 14000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    logp = np.log(100) + 0.00004 * i + 0.012 * np.sin(2 * np.pi * i / 900) + 0.012 * np.sin(2 * np.pi * i / 300 + 1.0)
    e = np.zeros(n)
    for k in range(1, n):
        e[k] = 0.9 * e[k - 1] + rng.normal(0, 0.0010)
    close = np.exp(logp + e)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0006, n)))
    vol = rng.uniform(80, 120, n)
    return _frame(open_, high, low, close, vol)


def tf_candles(df5: pd.DataFrame, limit: int = 500) -> dict[str, pd.DataFrame]:
    return {"5m": df5.tail(limit), "15m": resample_ohlcv(df5, "15min").tail(limit),
            "1h": resample_ohlcv(df5, "1h").tail(limit), "4h": resample_ohlcv(df5, "4h").tail(limit)}


def mirror(df5: pd.DataFrame, c: float = 400.0) -> pd.DataFrame:
    m = df5.copy()
    m["open"], m["high"], m["low"], m["close"] = c - df5["open"], c - df5["low"], c - df5["high"], c - df5["close"]
    return m


def _tail(sub: pd.DataFrame, L: float, a: float, p0: float, spike: float) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    closes = list(np.linspace(p0, L + 0.25 * a, 10) + rng.normal(0, 0.01 * a, 10))   # pullback to the level
    opens = [p0] + closes[:-1]
    highs = [max(o, c) + 0.03 * a for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.03 * a for o, c in zip(opens, closes)]
    vols = list(rng.uniform(80, 120, 10))
    # sweep candle (closes below the level), reclaim candle (high volume), drift up, confirmation candle
    seq = [
        (closes[-1], closes[-1] + 0.02 * a, L - 0.5 * a, L - 0.1 * a, 100 * spike * 0.6),
        (L - 0.1 * a, L + 0.2 * a, L - 0.6 * a, L + 0.15 * a, 100 * spike),
        (L + 0.15 * a, L + 0.30 * a, L + 0.10 * a, L + 0.25 * a, 110),
        (L + 0.25 * a, L + 0.40 * a, L + 0.20 * a, L + 0.35 * a, 105),
        (L + 0.35 * a, L + 0.48 * a, L + 0.30 * a, L + 0.42 * a, 100),
        (L + 0.42 * a, L + 0.50 * a, L + 0.36 * a, L + 0.46 * a, 100),
        (L + 0.40 * a, L + 0.66 * a, L + 0.30 * a, L + 0.62 * a, 120),
    ]
    for o, h, l, c, v in seq:
        opens.append(o); highs.append(h); lows.append(l); closes.append(c); vols.append(v)
    return _frame(opens, highs, lows, closes, vols, start=sub.index[-1] + pd.Timedelta(minutes=5))


def build_long_scenario(cfg: AppConfig | None = None, spike: float = 3.0, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Up-trending market that pulls back, sweeps a swing low and reclaims it. Returns (5m frame, info)."""
    key = ("long", spike, seed)
    if key in _CACHE:
        return _CACHE[key]
    cfg = cfg or default_cfg()
    if seed == 7 and spike == 3.0:                       # fast path: parameters found by the search below
        full = base_uptrend(seed=8)
        sub = full.iloc[:11208]
        a, price = 0.4581050488141728, float(sub["close"].iloc[-1])
        out = pd.concat([sub, _tail(sub, 156.0502469185889, a, price, spike)])
        ev = StrategyEngine(cfg.strategy, cfg.risk, cfg.ratio).evaluate(make_input(out, cfg))
        if ev.candidates["LONG"].outcome is Outcome.TRADE_SETUP:
            _CACHE[key] = (out, {"level": 156.0502469185889, "atr": a, "cut": 11208, "seed": 8})
            return _CACHE[key]
    for sd in range(seed, seed + 8):
        res = _search(cfg, spike, sd)
        if res:
            _CACHE[key] = res
            return res
    raise AssertionError("synthetic scenario search failed")


def _search(cfg: AppConfig, spike: float, seed: int):
    full = base_uptrend(seed=seed)
    for t in range(len(full) - 80, len(full) - 6000, -24):
        sub = full.iloc[:t]
        cs = tf_candles(sub)
        mtf0 = analyze_mtf(cs, cfg.strategy)
        if mtf0.bias != "bullish":
            continue
        a, price = mtf0.frames["15m"].atr, float(sub["close"].iloc[-1])
        tol = 0.5 * a
        lv = merge_levels(find_levels(cs["15m"], tol) + find_levels(cs["1h"], tol), tol)
        for l in [l for l in lv if l.low_touches > 0 and 0.5 * a <= price - l.price <= 1.4 * a]:
            out = pd.concat([sub, _tail(sub, l.price, a, price, spike)])
            if analyze_mtf(tf_candles(out), cfg.strategy).bias != "bullish":
                continue
            ev = StrategyEngine(cfg.strategy, cfg.risk, cfg.ratio).evaluate(make_input(out, cfg))
            if ev.candidates["LONG"].outcome is Outcome.TRADE_SETUP:      # fixture = a market where everything qualifies
                return out, {"level": l.price, "atr": a, "cut": t, "seed": seed}
    return None


def build_short_scenario(cfg: AppConfig | None = None, spike: float = 3.0, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    df, info = build_long_scenario(cfg, spike, seed)
    return mirror(df), {**info, "level": 400.0 - info["level"]}


def ratio_reading(value, now, kind=RatioKind.GLOBAL_ACCOUNT, age_s=60) -> RatioReading:
    return RatioReading(value, kind, now - timedelta(seconds=age_s) if age_s is not None else None, "test", "5m")


def make_input(df5: pd.DataFrame, cfg: AppConfig, ratio=None, heatmap=None, orderbook=None, price=None, now=None,
               oi_change=None, price_change=None, funding=None) -> StrategyInput:
    cs = tf_candles(df5, cfg.strategy.candle_limit)
    now = now or (df5["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2))
    return StrategyInput(symbol="TESTUSDT", now=now, price=float(df5["close"].iloc[-1]) if price is None else price,
                         candles=cs, ratio=ratio, heatmap=heatmap, orderbook=orderbook, oi_change_pct=oi_change,
                         price_change_pct=price_change, funding_rate=funding, volume_24h_usd=1e9)


def flatten_volume(df5: pd.DataFrame, n: int = 20, value: float = 100.0) -> pd.DataFrame:
    out = df5.copy()
    out.iloc[-n:, out.columns.get_loc("volume")] = value
    return out


def engine_for(cfg: AppConfig) -> StrategyEngine:
    return StrategyEngine(cfg.strategy, cfg.risk, cfg.ratio)


def strong_heatmap(cfg: AppConfig, df5: pd.DataFrame, above: bool = True):
    """Test fixture heatmap (clearly synthetic) with one big cluster on the chosen side of price."""
    price = float(df5["close"].iloc[-1])
    now = df5["close_time"].iloc[-1].to_pydatetime()
    cl = [LiquidationCluster(price * (1.012 if above else 0.988), 5e8, "short" if above else "long", now - timedelta(seconds=60), "test"),
          LiquidationCluster(price * (0.99 if above else 1.01), 5e7, "long" if above else "short", now - timedelta(seconds=60), "test")]
    return analyze_clusters(cl, price, df5.tail(300), now + timedelta(seconds=2), cfg.strategy, source="test")


def extend(df5: pd.DataFrame, n: int, per_bar: float) -> pd.DataFrame:
    """Append n continuation bars drifting by `per_bar` price units each (positive = up)."""
    last = float(df5["close"].iloc[-1])
    closes = last + per_bar * np.arange(1, n + 1)
    opens = np.r_[last, closes[:-1]]
    wick = abs(per_bar) * 0.2
    tail = _frame(opens, np.maximum(opens, closes) + wick, np.minimum(opens, closes) - wick, closes, np.full(n, 100.0),
                  start=df5.index[-1] + pd.Timedelta(minutes=5))
    return pd.concat([df5, tail])


class FakeConnector:
    """In-memory exchange connector backed by synthetic candles (tests only)."""
    name = "fake"

    def __init__(self, df5, ratio=None, fail=False, price=None):
        from decimal import Decimal
        from src.models import OrderBook, SymbolRules, TimedValue
        self.df5, self.ratio, self.fail, self._price = df5, ratio, fail, price
        self.rules = SymbolRules("TESTUSDT", Decimal("0.01"), Decimal("0.001"), Decimal("0.001"), Decimal("5"))
        self.calls = 0

    def _maybe_fail(self):
        self.calls += 1
        if self.fail:
            from src.exchanges.http import ExchangeError
            raise ExchangeError("simulated outage", 503)

    async def klines(self, symbol, interval, limit=500, start_ms=None, end_ms=None):
        self._maybe_fail()
        return tf_candles(self.df5, limit)[interval]

    async def ticker(self, symbol):
        self._maybe_fail()
        return (self._price or float(self.df5["close"].iloc[-1])), 5e9, self.df5["close_time"].iloc[-1]

    async def mark_and_funding(self, symbol):
        self._maybe_fail()
        from src.models import TimedValue
        return float(self.df5["close"].iloc[-1]), TimedValue(0.0, "fake", self.df5["close_time"].iloc[-1].to_pydatetime())

    async def funding_history(self, symbol, limit=100):
        self._maybe_fail()
        return pd.DataFrame({"rate": [0.0]}, index=[self.df5.index[-1]])

    async def open_interest_history(self, symbol, period="5m", limit=100):
        self._maybe_fail()
        idx = self.df5.index[-30:]
        return pd.DataFrame({"oi": np.linspace(1000, 1010, 30), "oi_value": 1.0}, index=idx)

    async def ratio_history(self, kind, symbol, period="5m", limit=100):
        self._maybe_fail()
        if self.ratio is None:
            from src.exchanges.base import NotSupported
            raise NotSupported("no ratio")
        idx = self.df5["close_time"].iloc[-12:]
        return pd.DataFrame({"ratio": [self.ratio] * 12}, index=idx)

    async def order_book(self, symbol, limit=100):
        self._maybe_fail()
        from src.models import OrderBook
        p = float(self.df5["close"].iloc[-1])
        return OrderBook([(p * 0.9999, 5000)], [(p * 1.0001, 5000)], self.df5["close_time"].iloc[-1].to_pydatetime(), "fake")

    async def symbol_rules(self, symbol):
        self._maybe_fail()
        return self.rules


class UniverseConnector(FakeConnector):
    """Fake multi-coin exchange: per-symbol candle frames + bulk tickers + universe discovery."""

    def __init__(self, frames: dict, volumes: dict | None = None, onboard_days: dict | None = None, ratio=None):
        first = next(iter(frames.values()))
        super().__init__(first, ratio=ratio)
        self.frames, self.volumes, self.onboard_days = frames, volumes or {}, onboard_days or {}
        self.kline_calls: dict = {}

    async def klines(self, symbol, interval, limit=500, start_ms=None, end_ms=None):
        self._maybe_fail()
        self.kline_calls[(symbol, interval)] = self.kline_calls.get((symbol, interval), 0) + 1
        return tf_candles(self.frames[symbol], limit)[interval]

    async def universe(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return [{"symbol": s, "onboard": now - timedelta(days=self.onboard_days.get(s, 400))} for s in self.frames]

    async def all_tickers(self):
        return pd.DataFrame({"last": [float(f["close"].iloc[-1]) for f in self.frames.values()],
                             "quote_volume": [self.volumes.get(s, 5e8) for s in self.frames],
                             "change_pct": 0.0}, index=list(self.frames))

    async def ticker(self, symbol):
        f = self.frames[symbol]
        return float(f["close"].iloc[-1]), self.volumes.get(symbol, 5e8), f["close_time"].iloc[-1]

    async def mark_and_funding(self, symbol):
        from src.models import TimedValue
        return float(self.frames[symbol]["close"].iloc[-1]), TimedValue(0.0, "fake", self.frames[symbol]["close_time"].iloc[-1].to_pydatetime())

    async def order_book(self, symbol, limit=100):
        from src.models import OrderBook
        f = self.frames[symbol]
        p = float(f["close"].iloc[-1])
        return OrderBook([(p * 0.9999, 5000)], [(p * 1.0001, 5000)], f["close_time"].iloc[-1].to_pydatetime(), "fake")


def noise_frame(n: int = 14000, seed: int = 3) -> pd.DataFrame:
    """Mean-reverting, trendless market (triage should skip it)."""
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for k in range(1, n):
        e[k] = 0.995 * e[k - 1] + rng.normal(0, 0.0006)
    c = 100 * np.exp(e)
    o = np.r_[c[0], c[:-1]]
    return _frame(o, np.maximum(o, c) * 1.0004, np.minimum(o, c) * 0.9996, c, rng.uniform(80, 120, n))
