"""Historical data download (paged) with a CSV cache under data/. Needs network access to the exchange."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import ROOT
from src.models import RatioKind
from src.utils import utcnow


async def fetch_klines_paged(connector, symbol: str, interval: str, days: float, page: int = 1500, now=None) -> pd.DataFrame:
    now = pd.Timestamp(now or utcnow())
    step = pd.Timedelta(minutes={"5m": 5, "15m": 15, "1h": 60}[interval])
    start = now - pd.Timedelta(days=days)
    cur, frames = start, []
    while cur < now:
        df = await connector.klines(symbol, interval, page, start_ms=int(cur.timestamp() * 1000))
        if df is None or df.empty:
            break
        frames.append(df)
        nxt = df.index[-1] + step
        if nxt <= cur:
            break
        cur = nxt
    if not frames:
        raise RuntimeError(f"no {interval} candles returned for {symbol}")
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].sort_index()


async def fetch_history_backwards(fetch, days: float, per_page: int = 500, step_minutes: int = 5, now=None) -> pd.DataFrame:
    """Page a `fetch(limit, end_ms)` endpoint backwards in time (ratio / OI history: exchanges keep ~30 days)."""
    now = pd.Timestamp(now or utcnow())
    stop = now - pd.Timedelta(days=days)
    end, frames = now, []
    while end > stop:
        df = await fetch(per_page, int(end.timestamp() * 1000))
        if df is None or df.empty:
            break
        frames.append(df)
        new_end = df.index[0] - pd.Timedelta(minutes=step_minutes)
        if new_end >= end:
            break
        end = new_end
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].sort_index().loc[stop:]


def cache_path(symbol: str, interval: str = "5m") -> Path:
    return ROOT / "data" / f"{symbol}_{interval}.csv"


def save_candles(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df
