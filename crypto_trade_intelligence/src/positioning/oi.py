"""Open-interest / price relationship (contextual evidence only, no causal claims)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.utils import is_finite


def pct_change_over(df: pd.DataFrame | None, col: str, minutes: int) -> float | None:
    if df is None or len(df) < 2 or col not in df.columns:
        return None
    last_t, last_v = df.index[-1], float(df[col].iloc[-1])
    target = last_t - pd.Timedelta(minutes=minutes)
    past = df.loc[:target]
    if past.empty or (target - past.index[-1]) > pd.Timedelta(minutes=max(minutes, 10)):
        return None
    base = float(past[col].iloc[-1])
    return (last_v / base - 1.0) * 100.0 if base > 0 else None


def classify(price_chg_pct: float | None, oi_chg_pct: float | None, flat: float = 0.15) -> str:
    """Return a descriptive label for the price/OI combination over the SAME window."""
    if not (is_finite(price_chg_pct) and is_finite(oi_chg_pct)):
        return "unavailable"
    p = 0 if abs(price_chg_pct) < flat else (1 if price_chg_pct > 0 else -1)
    o = 0 if abs(oi_chg_pct) < flat else (1 if oi_chg_pct > 0 else -1)
    return {
        (1, 1): "price up + OI up: possibly new positions entering (confirm with other data)",
        (-1, 1): "price down + OI up: possibly new positions entering during weakness",
        (1, -1): "price up + OI down: possibly short covering",
        (-1, -1): "price down + OI down: possibly long liquidation / position reduction",
    }.get((p, o), "price or OI roughly flat: no clear signal")


def oi_direction_score(direction_sign: int, price_chg: float | None, oi_chg: float | None, flat: float = 0.15) -> float | None:
    if not (is_finite(price_chg) and is_finite(oi_chg)):
        return None
    p = 0 if abs(price_chg) < flat else (1 if price_chg > 0 else -1)
    o = 0 if abs(oi_chg) < flat else (1 if oi_chg > 0 else -1)
    if p == 0 or o == 0:
        return 0.5
    with_trade = p == direction_sign
    if with_trade and o > 0:
        return 1.0     # new positions entering in the trade direction
    if with_trade and o < 0:
        return 0.5     # move on falling OI is weaker (possible covering)
    if not with_trade and o < 0:
        return 0.7     # counter-move on falling OI looks like a flush / reduction
    return 0.2         # counter-move on rising OI: positions being added against the trade


def oi_contradiction(direction_sign: int, price_chg: float | None, oi_chg: float | None, threshold_pct: float) -> bool:
    """Major contradiction: positions being added (OI rising materially) while price moves against the trade."""
    if not (is_finite(price_chg) and is_finite(oi_chg)):
        return False
    against = (price_chg < 0) if direction_sign > 0 else (price_chg > 0)
    return against and oi_chg >= threshold_pct
