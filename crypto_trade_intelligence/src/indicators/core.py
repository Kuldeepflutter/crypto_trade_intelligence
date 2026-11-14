"""Custom indicator implementations (no pandas-ta dependency; no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CANDLE_COLS = ["open", "high", "low", "close", "volume", "close_time"]


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, down = df["high"].diff(), -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored to the UTC day start."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.floor("D")
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Volume divided by the mean of the PREVIOUS n bars (current bar excluded)."""
    base = df["volume"].shift(1).rolling(n, min_periods=max(5, n // 2)).mean()
    return df["volume"] / base.replace(0, np.nan)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate lower-timeframe candles; returns only COMPLETE higher-timeframe bars."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "close_time": "max"}
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    span = pd.Timedelta(rule)
    last_close = df["close_time"].iloc[-1]
    return out[out.index + span - pd.Timedelta(milliseconds=1) <= last_close]


def validate_candles(df: pd.DataFrame | None, min_rows: int = 1) -> str | None:
    """Return a problem description, or None if the frame is usable."""
    if df is None or len(df) < min_rows:
        return f"need >= {min_rows} candles, have {0 if df is None else len(df)}"
    if any(c not in df.columns for c in CANDLE_COLS):
        return "missing candle columns"
    tail = df[["open", "high", "low", "close", "volume"]].tail(min(len(df), 250))
    if not np.isfinite(tail.to_numpy(dtype=float)).all():
        return "NaN/inf in recent candles"
    if (tail[["open", "high", "low", "close"]] <= 0).any().any():
        return "non-positive price in recent candles"
    if (tail["high"] < tail["low"]).any():
        return "high < low in recent candles"
    if not df.index.is_monotonic_increasing:
        return "candles not time-ordered"
    return None
