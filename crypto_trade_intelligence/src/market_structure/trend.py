"""Multi-timeframe trend classification (EMA stack + structure + ADX + VWAP + volume)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import StrategyConfig
from src.indicators.core import adx, atr, ema, validate_candles, volume_ratio, vwap
from src.market_structure.swings import StructureResult, analyze_structure
from src.models import TrendLabel


@dataclass
class TimeframeTrend:
    timeframe: str
    label: TrendLabel
    close: float = float("nan")
    atr: float = float("nan")
    adx: float = float("nan")
    ema_score: float = 0.0
    struct_dir: int = 0
    vwap_dir: int = 0
    score: float = 0.0
    volume_ratio: float = float("nan")
    structure: StructureResult = field(default_factory=StructureResult)
    note: str = ""


@dataclass
class MTFTrend:
    frames: dict[str, TimeframeTrend]
    bias: str                      # bullish | bearish | conflict | neutral | unknown
    bias_quality: float            # 0..1 (strength of the higher-timeframe agreement)
    notes: list[str] = field(default_factory=list)

    def labels(self) -> dict[str, str]:
        return {tf: t.label.value for tf, t in self.frames.items()}


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def analyze_timeframe(tf: str, df: pd.DataFrame | None, cfg: StrategyConfig) -> TimeframeTrend:
    need = cfg.ema_slow + 10
    problem = validate_candles(df, need)
    if problem:
        return TimeframeTrend(tf, TrendLabel.UNCERTAIN, note=f"insufficient/invalid data: {problem}")
    close = df["close"]
    e20, e50, e200 = ema(close, cfg.ema_fast), ema(close, cfg.ema_mid), ema(close, cfg.ema_slow)
    a, adx_s, vw, vr = atr(df, cfg.atr_period), adx(df, cfg.adx_period), vwap(df), volume_ratio(df, cfg.volume_ma)
    c = float(close.iloc[-1])
    vals = [float(x.iloc[-1]) for x in (e20, e50, e200, a, adx_s)]
    if not np.isfinite(vals).all():
        return TimeframeTrend(tf, TrendLabel.UNCERTAIN, note="indicators not finite")
    v20, v50, v200, atr_v, adx_v = vals
    ema_score = (_sign(c - v20) + _sign(v20 - v50) + _sign(v50 - v200)) / 3.0
    st = analyze_structure(df, cfg.swing_left, cfg.swing_right, cfg.event_lookback)
    struct_dir = {"up": 1, "down": -1}.get(st.trend, 0)
    vw_last = float(vw.iloc[-1]) if np.isfinite(vw.iloc[-1]) else c
    vwap_dir = _sign(c - vw_last)
    score = 0.45 * ema_score + 0.40 * struct_dir + 0.15 * vwap_dir
    note = ""
    if ema_score * struct_dir < 0:
        label, note = TrendLabel.UNCERTAIN, "EMA stack and market structure disagree"
    elif abs(score) < 0.25 or (adx_v < cfg.adx_range and abs(score) < 0.6):
        label = TrendLabel.RANGE
    else:
        strong = abs(score) >= 0.6 and adx_v >= cfg.adx_strong
        up = score > 0
        label = (TrendLabel.STRONG_UP if strong else TrendLabel.WEAK_UP) if up else \
                (TrendLabel.STRONG_DOWN if strong else TrendLabel.WEAK_DOWN)
    return TimeframeTrend(tf, label, c, atr_v, adx_v, ema_score, struct_dir, vwap_dir, score,
                          float(vr.iloc[-1]) if np.isfinite(vr.iloc[-1]) else float("nan"), st, note)


def analyze_mtf(candles: dict[str, pd.DataFrame], cfg: StrategyConfig) -> MTFTrend:
    frames = {tf: analyze_timeframe(tf, candles.get(tf), cfg) for tf in ("5m", "15m", "1h", "4h")}
    h1, h4 = frames["1h"], frames["4h"]
    notes: list[str] = []
    if h1.atr != h1.atr or h4.atr != h4.atr:          # NaN -> data problem on a bias timeframe
        notes += [f"{t.timeframe}: {t.note}" for t in (h1, h4) if t.note]
        return MTFTrend(frames, "unknown", 0.0, notes)
    d1, d4 = h1.label.direction, h4.label.direction
    if h1.label is TrendLabel.UNCERTAIN or h4.label is TrendLabel.UNCERTAIN:
        notes.append("1h/4h indicators conflict internally")
        return MTFTrend(frames, "conflict", 0.0, notes)
    if d1 and d4 and d1 != d4:
        notes.append(f"1h ({h1.label.value}) vs 4h ({h4.label.value}) conflict")
        return MTFTrend(frames, "conflict", 0.0, notes)
    if d1 and d1 == d4:
        q = 1.0 if (h1.label.is_strong and h4.label.is_strong) else (0.8 if (h1.label.is_strong or h4.label.is_strong) else 0.6)
        return MTFTrend(frames, "bullish" if d1 > 0 else "bearish", q, notes)
    if d1 and not d4:
        notes.append("4h is range-bound: partial higher-timeframe confirmation only")
        return MTFTrend(frames, "bullish" if d1 > 0 else "bearish", 0.45, notes)
    notes.append("1h is range-bound or neutral")
    return MTFTrend(frames, "neutral", 0.0, notes)


def reversal_confirmed(direction_sign: int, mtf: MTFTrend, cfg: StrategyConfig) -> tuple[bool, str]:
    """A clearly defined reversal: 1h CHoCH in the trade direction + 15m structure agrees + 4h not strongly opposed."""
    h1, m15, h4 = mtf.frames["1h"], mtf.frames["15m"], mtf.frames["4h"]
    want = "bullish" if direction_sign > 0 else "bearish"
    choch_ok = h1.structure.choch == want and (h1.structure.choch_bars_ago or 0) <= cfg.reversal_lookback_bars
    m15_ok = m15.struct_dir == direction_sign or m15.structure.choch == want or m15.structure.bos == want
    h4_ok = not (h4.label.is_strong and h4.label.direction == -direction_sign)
    ok = choch_ok and m15_ok and h4_ok
    return ok, ("1h CHoCH + 15m agreement" if ok else "no confirmed reversal")
