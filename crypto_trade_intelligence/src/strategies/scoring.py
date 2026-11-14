"""Configurable 0-100 setup score. Hard filters run first; the score is NOT a calibrated win probability."""
from __future__ import annotations

import numpy as np

from config import StrategyConfig
from src.market_structure.trend import MTFTrend
from src.models import Direction, OrderBookMetrics
from src.positioning.oi import oi_direction_score
from src.positioning.ratio import ratio_context_score
from src.utils import is_finite


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def compute_components(d: Direction, mtf: MTFTrend, htf_quality: float, vol_ratio: float | None, heat_score: float | None,
                       oi_score_inputs: tuple[float | None, float | None], ratio: float | None, funding: float | None,
                       ob: OrderBookMetrics | None, rr_net: float, data_quality: float, cfg: StrategyConfig) -> dict[str, dict]:
    s = d.sign
    m15, m5, h1 = mtf.frames["15m"], mtf.frames["5m"], mtf.frames["1h"]
    want = "bullish" if s > 0 else "bearish"
    struct = 0.4 + 0.3 * (m15.struct_dir == s) + 0.2 * (m5.structure.bos == want or m5.structure.choch == want) + 0.1 * (h1.struct_dir == s)
    vol = clamp01((vol_ratio or 0) / (2 * cfg.min_volume_ratio)) if is_finite(vol_ratio) else None
    oi = oi_direction_score(s, *oi_score_inputs)
    fund = None
    if is_finite(funding):
        f = funding * s
        fund = float(np.interp(f, [-cfg.funding_extreme, 0, cfg.funding_extreme, 2 * cfg.funding_extreme], [1.0, 0.8, 0.4, 0.1]))
    book = None
    if ob is not None:
        book = clamp01(0.5 + 0.5 * (ob.imbalance * s) / 0.4)
    rr = clamp01(0.5 + 0.5 * (rr_net - cfg.min_rr) / cfg.min_rr)
    raw = {
        "htf_trend": (htf_quality, "higher-timeframe agreement quality"),
        "structure": (struct, "15m/5m/1h market structure alignment"),
        "liquidation": (heat_score, "estimated liquidation-cluster context"),
        "volume": (vol, f"trigger volume ratio {vol_ratio:.2f}x" if is_finite(vol_ratio) else "volume ratio n/a"),
        "oi": (oi, "OI vs price behaviour (contextual)"),
        "ratio": (ratio_context_score(s, ratio, cfg.ratio_crowded), f"long/short ratio {ratio}" if is_finite(ratio) else "ratio n/a"),
        "funding": (fund, "funding-rate crowding"),
        "orderbook": (book, "order-book imbalance"),
        "rr": (rr, f"net R/R {rr_net:.2f}"),
        "data_quality": (data_quality, "share of optional inputs present and fresh"),
    }
    return {k: {"score": (None if v is None else clamp01(v)), "weight": cfg.weights.get(k, 0), "note": n} for k, (v, n) in raw.items()}


def aggregate(components: dict[str, dict]) -> float:
    """Weighted average over components that have data (missing ones are excluded, not scored as zero)."""
    num = sum(c["score"] * c["weight"] for c in components.values() if c["score"] is not None)
    den = sum(c["weight"] for c in components.values() if c["score"] is not None)
    return 0.0 if den <= 0 else 100.0 * num / den
