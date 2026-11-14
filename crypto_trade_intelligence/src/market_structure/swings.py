"""Swing points, market structure (HH/HL/LH/LL), BOS / CHoCH and support/resistance levels.

Swings are fractal pivots confirmed only after `right` further bars, so nothing here peeks
into the future: a swing at index i is only usable once bar i+right has closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Swing:
    idx: int
    time: pd.Timestamp
    price: float
    kind: str  # "high" | "low"


@dataclass
class StructureResult:
    trend: str = "unknown"                 # up | down | range | unknown
    last_high: Swing | None = None
    last_low: Swing | None = None
    bos: str | None = None                 # "bullish" | "bearish"
    bos_bars_ago: int | None = None
    choch: str | None = None
    choch_bars_ago: int | None = None
    swings: list[Swing] = field(default_factory=list)


@dataclass
class Level:
    price: float
    touches: int
    high_touches: int      # swing highs in the cluster (was resistance)
    low_touches: int       # swing lows in the cluster (was support)
    last_idx: int


def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Swing]:
    h, l = df["high"], df["low"]
    prev_max = h.shift(1).rolling(left).max()
    next_max = h.iloc[::-1].shift(1).rolling(right).max().iloc[::-1]
    prev_min = l.shift(1).rolling(left).min()
    next_min = l.iloc[::-1].shift(1).rolling(right).min().iloc[::-1]
    is_high = ((h > prev_max) & (h >= next_max)).to_numpy()
    is_low = ((l < prev_min) & (l <= next_min)).to_numpy()
    out: list[Swing] = []
    idx = df.index
    for i in np.flatnonzero(is_high | is_low):
        if is_high[i]:
            out.append(Swing(int(i), idx[i], float(h.iloc[i]), "high"))
        if is_low[i]:
            out.append(Swing(int(i), idx[i], float(l.iloc[i]), "low"))
    return out


def _classify(highs: list[Swing], lows: list[Swing]) -> str:
    if len(highs) < 2 or len(lows) < 2:
        return "unknown"
    h1, h2 = highs[-2].price, highs[-1].price
    l1, l2 = lows[-2].price, lows[-1].price
    if h2 > h1 and l2 > l1:
        return "up"
    if h2 < h1 and l2 < l1:
        return "down"
    return "range"


def _first_close_beyond(closes: np.ndarray, start: int, level: float, above: bool) -> int | None:
    for j in range(start, len(closes)):
        if (closes[j] > level) if above else (closes[j] < level):
            return j
    return None


def analyze_structure(df: pd.DataFrame, left: int = 3, right: int = 3, event_lookback: int = 12) -> StructureResult:
    swings = find_swings(df, left, right)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    res = StructureResult(trend=_classify(highs, lows), swings=swings)
    res.last_high = highs[-1] if highs else None
    res.last_low = lows[-1] if lows else None
    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)
    events: list[tuple[int, str]] = []
    if res.last_high is not None:
        j = _first_close_beyond(closes, res.last_high.idx + right + 1, res.last_high.price, True)
        if j is not None:
            events.append((j, "bullish"))
    if res.last_low is not None:
        j = _first_close_beyond(closes, res.last_low.idx + right + 1, res.last_low.price, False)
        if j is not None:
            events.append((j, "bearish"))
    if events:
        j, side = max(events)
        ago = n - 1 - j
        if ago <= event_lookback:
            reversal = (side == "bullish" and res.trend == "down") or (side == "bearish" and res.trend == "up")
            if reversal:
                res.choch, res.choch_bars_ago = side, ago
            else:
                res.bos, res.bos_bars_ago = side, ago
    return res


def find_levels(df: pd.DataFrame, tol: float, left: int = 3, right: int = 3, max_bars: int = 250) -> list[Level]:
    """Cluster confirmed swing prices that lie within `tol` (price units) of each other."""
    sw = [s for s in find_swings(df.tail(max_bars), left, right)]
    if not sw:
        return []
    sw.sort(key=lambda s: s.price)
    groups: list[list[Swing]] = [[sw[0]]]
    for s in sw[1:]:
        if s.price - groups[-1][-1].price <= tol:
            groups[-1].append(s)
        else:
            groups.append([s])
    return [Level(float(np.mean([s.price for s in g])), len(g),
                  sum(s.kind == "high" for s in g), sum(s.kind == "low" for s in g),
                  max(s.idx for s in g)) for g in groups]


def merge_levels(levels: list[Level], tol: float) -> list[Level]:
    if not levels:
        return []
    levels = sorted(levels, key=lambda lv: lv.price)
    out = [levels[0]]
    for lv in levels[1:]:
        cur = out[-1]
        if lv.price - cur.price <= tol:
            t = cur.touches + lv.touches
            out[-1] = Level((cur.price * cur.touches + lv.price * lv.touches) / t, t,
                            cur.high_touches + lv.high_touches, cur.low_touches + lv.low_touches,
                            max(cur.last_idx, lv.last_idx))
        else:
            out.append(lv)
    return out
