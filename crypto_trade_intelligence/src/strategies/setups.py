"""Price-action triggers: liquidity sweep + reclaim, breakout-retest, level bounce.

All logic is written once for the LONG case on prices; SHORT runs on a price-mirrored copy
(prices negated, high/low swapped) and the result is mapped back. A heatmap cluster is NOT a
trigger - these triggers come from candles only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from config import StrategyConfig
from src.indicators.core import volume_ratio
from src.market_structure.swings import find_levels, merge_levels
from src.models import Direction

KIND_PRIORITY = {"liquidity_sweep": 0, "breakout_retest": 1, "level_bounce": 2}


@dataclass(frozen=True)
class Trigger:
    kind: str
    status: str               # "valid" | "pending"
    level: float
    entry: float
    invalidation: float       # structural invalidation (before volatility buffer)
    confirm_extreme: float    # high (LONG) / low (SHORT) of the latest candle
    volume_ratio: float
    note: str
    timeframe: str = "5m"
    reclaim: bool = False


def _mirror(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()
    m["open"], m["high"], m["low"], m["close"] = -df["open"], -df["low"], -df["high"], -df["close"]
    return m


def _unmirror(t: Trigger) -> Trigger:
    return replace(t, level=-t.level, entry=-t.entry, invalidation=-t.invalidation, confirm_extreme=-t.confirm_extreme)


def _confirm(c5: pd.DataFrame) -> bool:
    last, prev = c5.iloc[-1], c5.iloc[-2]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return False
    lower_wick = (min(last["open"], last["close"]) - last["low"]) / rng
    return bool(last["close"] > last["open"] and last["close"] > prev["close"] and
                (last["close"] > prev["high"] or lower_wick >= 0.4))


def _long_triggers(c5: pd.DataFrame, c15: pd.DataFrame, c1h: pd.DataFrame, atr: float, cfg: StrategyConfig) -> list[Trigger]:
    out: list[Trigger] = []
    price = float(c5["close"].iloc[-1])
    tol = 0.5 * atr
    levels = merge_levels(find_levels(c15, tol, cfg.swing_left, cfg.swing_right) +
                          find_levels(c1h, tol, cfg.swing_left, cfg.swing_right), tol)
    vr5 = volume_ratio(c5, cfg.volume_ma)
    n = cfg.sweep_lookback_bars
    seg = c5.iloc[-(n + 3):]
    lows, highs, closes = seg["low"].to_numpy(float), seg["high"].to_numpy(float), seg["close"].to_numpy(float)
    vols = vr5.iloc[-(n + 3):].to_numpy(float)
    eps, last_high = 0.02 * atr, float(c5["high"].iloc[-1])
    recent_vr = float(np.nanmax(vr5.iloc[-3:])) if vr5.iloc[-3:].notna().any() else float("nan")

    # 1) liquidity sweep of a swing low, then reclaim
    for lv in levels:
        L = lv.price
        if lv.low_touches == 0 or L > price + 3 * atr:
            continue
        sw = np.flatnonzero(lows < L - eps)
        if sw.size == 0:
            continue
        k0, k1 = int(sw[0]), int(sw[-1])
        if k0 == 0 or not (closes[:k0] > L).any():
            continue                                     # the level was not holding before the sweep
        extreme = float(lows[k0:].min())
        if L - extreme > 1.5 * atr:
            continue                                     # too deep: a break, not a sweep
        j = next((i for i in range(k0, min(len(closes), k1 + 4)) if closes[i] > L), None)
        if j is None:
            if k1 >= len(closes) - 3 and price < L and (L - price) <= atr:
                out.append(Trigger("liquidity_sweep", "pending", L, price, extreme, last_high, recent_vr,
                                   f"swing low {L:.6g} swept; waiting for a close back above"))
            continue
        if not (closes[j:] > L).all() or price - L > 2 * atr:
            continue
        vr = float(np.nanmax(vols[k0:j + 1])) if np.isfinite(vols[k0:j + 1]).any() else float("nan")
        out.append(Trigger("liquidity_sweep", "valid", L, price, extreme, last_high, vr,
                           f"swept swing low {L:.6g} (depth {(L - extreme) / atr:.2f} ATR) and reclaimed", reclaim=True))

    # 2) breakout of resistance on 15m, now retested from above
    m = cfg.retest_lookback_bars
    c15c = c15["close"].to_numpy(float)
    last6 = c5.iloc[-6:]
    if len(c15c) > m + 1:
        w, prev = c15c[-m:], c15c[-m - 1:-1]
        for lv in levels:
            R = lv.price
            if lv.high_touches == 0 or not (-0.3 * atr <= price - R <= cfg.proximity_atr * atr):
                continue
            if np.flatnonzero((w > R + 0.1 * atr) & (prev <= R)).size == 0:
                continue
            if (last6["close"] < R - 0.3 * atr).any():
                continue
            ok = _confirm(c5)
            out.append(Trigger("breakout_retest", "valid" if ok else "pending", R, price,
                               min(float(last6["low"].min()), R), last_high, recent_vr,
                               f"retest of broken resistance {R:.6g}" + ("" if ok else " (no confirmation candle yet)")))

    # 3) bounce from support with at least two touches
    for lv in levels:
        S = lv.price
        if lv.touches < 2 or lv.low_touches == 0 or not (-0.2 * atr <= price - S <= cfg.proximity_atr * atr):
            continue
        if (last6["close"] < S - 0.3 * atr).any():
            continue
        ok = _confirm(c5)
        out.append(Trigger("level_bounce", "valid" if ok else "pending", S, price,
                           min(S, float(c5["low"].iloc[-12:].min())), last_high, recent_vr,
                           f"support {S:.6g} ({lv.touches} touches)" + ("" if ok else " (no confirmation candle yet)")))
    return out


def detect_triggers(direction: Direction, candles: dict[str, pd.DataFrame], atr15: float, cfg: StrategyConfig) -> list[Trigger]:
    c5, c15, c1h = (candles[k].tail(300) for k in ("5m", "15m", "1h"))
    if direction is Direction.SHORT:
        c5, c15, c1h = _mirror(c5), _mirror(c15), _mirror(c1h)
    trig = _long_triggers(c5, c15, c1h, atr15, cfg)
    if direction is Direction.SHORT:
        trig = [_unmirror(t) for t in trig]
    return sorted(trig, key=lambda t: (t.status != "valid", KIND_PRIORITY[t.kind], -(t.volume_ratio if np.isfinite(t.volume_ratio) else 0)))
