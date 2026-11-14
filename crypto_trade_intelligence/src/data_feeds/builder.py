"""Turn a MarketSnapshot into a StrategyInput (derived metrics, no fabricated values)."""
from __future__ import annotations

import pandas as pd

from config import AppConfig
from src.data_feeds.orderbook import orderbook_metrics
from src.liquidation.heatmap import HeatmapContext, analyze_clusters, unavailable
from src.models import MarketSnapshot, RatioKind
from src.positioning.oi import pct_change_over
from src.positioning.ratio import latest_reading, ratio_changes
from src.strategies.engine import StrategyInput


def _close_at(c5: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    seg = c5.loc[c5["close_time"] <= ts]
    return float(seg["close"].iloc[-1]) if len(seg) else None


def build_input(snap: MarketSnapshot, cfg: AppConfig, previous_clusters=None) -> StrategyInput:
    kind = RatioKind(cfg.ratio.kind)
    hist = snap.ratio_history.get(kind)
    reading = latest_reading(hist, kind, f"{cfg.exchange.name}:{kind.value}", cfg.ratio.period)
    c5 = snap.candles.get("5m")
    oi_chg = pct_change_over(snap.oi_history, "oi", cfg.strategy.oi_window_minutes)
    price_chg = None
    if oi_chg is not None and c5 is not None and len(c5):
        end = snap.oi_history.index[-1]
        p1, p0 = _close_at(c5, end), _close_at(c5, end - pd.Timedelta(minutes=cfg.strategy.oi_window_minutes))
        price_chg = (p1 / p0 - 1) * 100 if p0 and p1 else None
    vol_chg = None
    if c5 is not None and len(c5) >= 24:
        a, b = float(c5["volume"].iloc[-12:].sum()), float(c5["volume"].iloc[-24:-12].sum())
        vol_chg = (a / b - 1) * 100 if b > 0 else None
    heat: HeatmapContext
    if snap.price and snap.liquidation_clusters:
        heat = analyze_clusters(snap.liquidation_clusters, snap.price, c5, snap.now, cfg.strategy, previous_clusters, snap.liquidation_source)
    else:
        heat = unavailable(snap.errors.get("liquidation", "no liquidation provider configured"), snap.liquidation_source)
    reduced = []
    if not heat.available and cfg.strategy.liquidation_mode == "optional":
        reduced.append("liquidation heatmap unavailable")
    if heat.available and str(snap.liquidation_source).startswith("public_model"):
        reduced.append("liquidation clusters are a keyless public-data MODEL (assumption-based, lower confidence)")
    return StrategyInput(
        symbol=snap.symbol, now=snap.now, price=snap.price, candles=snap.candles, ratio=reading,
        ratio_changes=ratio_changes(hist, snap.now), oi_change_pct=oi_chg, price_change_pct=price_chg,
        funding_rate=snap.funding.value if snap.funding else None, volume_change_pct=vol_chg,
        orderbook=orderbook_metrics(snap.orderbook, cfg.strategy.orderbook_depth_pct), heatmap=heat,
        volume_24h_usd=snap.volume_24h_usd, tick_size=snap.rules.tick_size if snap.rules else None,
        reduced_features=reduced)
