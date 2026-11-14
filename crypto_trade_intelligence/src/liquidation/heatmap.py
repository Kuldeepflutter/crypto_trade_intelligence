"""Liquidation-cluster analysis.

Clusters here are ESTIMATES supplied by a heatmap provider. They are never treated as a guaranteed
target and are kept separate from actual liquidation events (see models.LiquidationEvent).
Nothing in this module can, on its own, trigger a trade: the strategy engine only uses it as one
mandatory confirmation among several.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from config import StrategyConfig
from src.models import Direction, LiquidationCluster


@dataclass
class RankedCluster:
    cluster: LiquidationCluster
    distance_pct: float
    above: bool
    swept: bool | None            # None = cannot be determined
    reclaimed: bool = False       # swept below, then price closed back above (or mirror above)
    weakened: bool | None = None  # vs previous snapshot (None = no previous data)
    rank: float = 0.0
    age_seconds: float | None = None


@dataclass
class HeatmapContext:
    available: bool
    reason: str = ""
    source: str = "none"
    timestamp: datetime | None = None
    clusters: list[RankedCluster] = field(default_factory=list)

    def intact(self, above: bool) -> list[RankedCluster]:
        return [c for c in self.clusters if c.above == above and c.swept is False]

    @property
    def nearest_above(self) -> RankedCluster | None:
        c = self.intact(True)
        return min(c, key=lambda x: x.distance_pct) if c else None

    @property
    def nearest_below(self) -> RankedCluster | None:
        c = self.intact(False)
        return min(c, key=lambda x: x.distance_pct) if c else None

    def strongest(self, above: bool) -> RankedCluster | None:
        c = self.intact(above)
        return max(c, key=lambda x: x.rank) if c else None


def unavailable(reason: str, source: str = "none") -> HeatmapContext:
    return HeatmapContext(False, reason, source)


def analyze_clusters(clusters: list[LiquidationCluster] | None, price: float, candles_5m: pd.DataFrame | None,
                     now: datetime, cfg: StrategyConfig, previous: list[LiquidationCluster] | None = None,
                     source: str = "provider") -> HeatmapContext:
    if not clusters:
        return unavailable("no liquidation cluster data available", source)
    ts = max((c.timestamp for c in clusters if c.timestamp), default=None)
    if ts is None:
        return unavailable("cluster data has no timestamp (freshness cannot be verified)", source)
    age = (now - ts).total_seconds()
    if age > cfg.heatmap_max_age_seconds:
        return unavailable(f"cluster data stale ({age:.0f}s > {cfg.heatmap_max_age_seconds}s)", source)
    max_notional = max((c.notional_usd for c in clusters if math.isfinite(c.notional_usd)), default=0.0)
    if max_notional <= 0 or not math.isfinite(price) or price <= 0:
        return unavailable("cluster notional or price invalid", source)

    ranked: list[RankedCluster] = []
    for c in clusters:
        if not (math.isfinite(c.price) and c.price > 0 and math.isfinite(c.notional_usd) and c.notional_usd > 0):
            continue
        dist = abs(c.price - price) / price * 100
        above = c.price > price
        swept, reclaimed = _swept_status(c, above, candles_5m, price)
        weakened = None
        if previous:
            prev = [p for p in previous if abs(p.price - c.price) / c.price < 0.001]
            if prev:
                weakened = c.notional_usd < 0.7 * prev[0].notional_usd
        c_age = (now - c.timestamp).total_seconds() if c.timestamp else None
        fresh = 1.0 if c_age is None else max(0.5, 1.0 - 0.5 * c_age / cfg.heatmap_max_age_seconds)
        decay = math.exp(-dist / cfg.heatmap_decay_pct)
        rank = (c.notional_usd / max_notional) * decay * fresh * (0.0 if swept else 1.0) * (0.5 if weakened else 1.0)
        if dist > cfg.heatmap_max_distance_pct:
            rank = 0.0
        ranked.append(RankedCluster(c, dist, above, swept, reclaimed, weakened, rank, c_age))
    ranked.sort(key=lambda r: -r.rank)
    return HeatmapContext(True, "ok", source, ts, ranked)


def _swept_status(c: LiquidationCluster, above: bool, candles: pd.DataFrame | None, price: float) -> tuple[bool | None, bool]:
    """Has price traded through the cluster since the data timestamp, and was it reclaimed/rejected?"""
    if candles is None or candles.empty or c.timestamp is None:
        return None, False
    seg = candles.loc[candles["close_time"] >= pd.Timestamp(c.timestamp)]
    if seg.empty:
        return False, False
    if above:
        hit = seg["high"] >= c.price
        swept = bool(hit.any())
        reclaimed = swept and price < c.price     # swept up, now back below -> rejected
    else:
        hit = seg["low"] <= c.price
        swept = bool(hit.any())
        reclaimed = swept and price > c.price     # swept down, now back above -> reclaimed
    return swept, reclaimed


def heatmap_support(direction: Direction, ctx: HeatmapContext | None, recent_sweep_reclaim: bool = False) -> tuple[float | None, bool, str]:
    """Return (score 0..1 or None if unavailable, supports?, explanation).

    Supports a LONG when intact liquidity above (target) outweighs intact liquidity below (risk of
    being pulled down) OR a downside cluster was just swept and reclaimed. Mirrored for SHORT.
    """
    if ctx is None or not ctx.available:
        return None, False, f"liquidation data unavailable ({ctx.reason if ctx else 'not provided'})"
    long = direction is Direction.LONG
    target = ctx.strongest(above=long)
    risk = ctx.strongest(above=not long)
    t_rank = target.rank if target else 0.0
    r_rank = risk.rank if risk else 0.0
    reclaim = any(c.reclaimed for c in ctx.clusters if c.above != long and c.swept)
    score = 0.5 + 0.5 * (t_rank - r_rank)
    if reclaim or recent_sweep_reclaim:
        score += 0.25
    score = max(0.0, min(1.0, score))
    if t_rank == 0 and r_rank == 0 and not reclaim:
        return 0.0, False, "no relevant intact liquidation clusters near price"
    supports = score >= 0.5 and (t_rank > 0 or reclaim)
    why = (f"target cluster rank {t_rank:.2f} vs opposing {r_rank:.2f}"
           + (", opposing cluster swept & reclaimed" if reclaim else "")
           + " (estimated clusters, not guaranteed targets)")
    return score, supports, why
