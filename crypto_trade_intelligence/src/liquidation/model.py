"""KEYLESS liquidation-cluster ESTIMATE built only from public Binance data.

THIS IS A MODEL, NOT VENDOR DATA. Public exchanges do not publish where open positions would be
liquidated, so this reconstructs a plausible distribution under explicit assumptions:

  1. Every increase in open interest (OI) is new exposure opened at that bar's typical price.
  2. New exposure is split into longs / shorts using the public long/short ACCOUNT ratio of that
     time (accounts are not notional - this is an approximation).
  3. Leverage follows `leverage_mix` (an assumption, configurable).  Liquidation price =
     entry * (1 -/+ 1/leverage +/- maintenance_margin_rate)   (isolated-margin approximation).
  4. When OI falls, all existing exposure is scaled down proportionally (positions closing).
  5. Exposure whose liquidation price was traded through after it was opened is removed
     (already liquidated).  What remains is the "still-open" estimate, bucketed by price.

Not modelled: positions opened before the window, cross-margin, funding, partial deleveraging,
hidden orders. Treat clusters as lower-confidence context. They are labelled estimated=True and
the source string says so; the strategy uses them only as ONE confirmation among several.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import PublicModelConfig
from src.models import LiquidationCluster, RatioKind
from src.utils import utcnow

MODEL_SOURCE = "public_model:binance[OI x L/S-ratio x leverage-mix ASSUMPTIONS - estimate, not vendor data]"


@dataclass
class ModelResult:
    clusters: list[LiquidationCluster]
    reason: str = "ok"
    info: dict = field(default_factory=dict)


def estimate_clusters(candles: pd.DataFrame | None, oi: pd.DataFrame | None, ratio: pd.DataFrame | None,
                      cfg: PublicModelConfig, maintenance_margin_rate: float, period_minutes: int) -> ModelResult:
    if candles is None or len(candles) < 50:
        return ModelResult([], "not enough candles for the liquidation model")
    if oi is None or oi.empty:
        return ModelResult([], "open-interest history unavailable (model needs it)")
    if ratio is None or ratio.empty:
        return ModelResult([], "long/short ratio history unavailable (cannot split OI into longs and shorts)")
    tol = pd.Timedelta(minutes=period_minutes)
    ct = pd.DatetimeIndex(candles["close_time"])
    d = pd.DataFrame({
        "oi": oi["oi"].reindex(ct, method="nearest", tolerance=tol).to_numpy(),
        "ratio": ratio["ratio"].reindex(ct, method="nearest", tolerance=tol).to_numpy(),
        "high": candles["high"].to_numpy(), "low": candles["low"].to_numpy(),
        "tp": ((candles["high"] + candles["low"] + candles["close"]) / 3).to_numpy()}, index=ct)
    cover = float(d[["oi", "ratio"]].notna().all(axis=1).mean())
    if cover < cfg.min_coverage:
        return ModelResult([], f"OI/ratio history covers only {cover:.0%} of the candle window (< {cfg.min_coverage:.0%})")
    d = d.ffill(limit=3).dropna()
    if len(d) < 50:
        return ModelResult([], "too few aligned OI/ratio bars")

    levs = np.array(sorted(float(k) for k in cfg.leverage_mix))
    w = np.array([float(cfg.leverage_mix.get(k, cfg.leverage_mix.get(str(k), cfg.leverage_mix.get(int(k), 0)))) for k in levs])
    w = w / w.sum()
    cap = len(d) * len(levs) * 2
    E, Q, LQ, LG = np.zeros(cap), np.zeros(cap), np.zeros(cap), np.zeros(cap, bool)
    n = 0
    added = 0.0
    oi_v, ratio_v, hi, lo, tp = (d[c].to_numpy(float) for c in ("oi", "ratio", "high", "low", "tp"))
    for i in range(1, len(d)):
        if n:                                               # 1) exposure whose liquidation price this bar traded through is gone
            dead = (LG[:n] & (LQ[:n] >= lo[i])) | (~LG[:n] & (LQ[:n] <= hi[i]))
            Q[:n][dead] = 0.0
        delta = oi_v[i] - oi_v[i - 1]
        if delta < 0 and oi_v[i - 1] > 0 and n:             # 4) positions closing: scale everything down proportionally
            Q[:n] *= oi_v[i] / oi_v[i - 1]
        elif delta > 0 and ratio_v[i] > 0:                  # 2)+3) new exposure, split by ratio and leverage mix
            long_share = ratio_v[i] / (1.0 + ratio_v[i])
            added += delta * tp[i]
            for L, wt in zip(levs, w):
                E[n], Q[n], LQ[n], LG[n] = tp[i], delta * long_share * wt, tp[i] * (1 - 1 / L + maintenance_margin_rate), True
                n += 1
                E[n], Q[n], LQ[n], LG[n] = tp[i], delta * (1 - long_share) * wt, tp[i] * (1 + 1 / L - maintenance_margin_rate), False
                n += 1
    alive = Q[:n] > 1e-12
    if not alive.any():
        return ModelResult([], "no surviving modelled exposure (no OI growth in window or all liquidated)", {"added_usd": added})
    notion, liq, is_long = (Q[:n] * E[:n])[alive], LQ[:n][alive], LG[:n][alive]
    ref = float(candles["close"].iloc[-1])
    width = ref * cfg.bucket_pct / 100.0
    df = pd.DataFrame({"bucket": np.floor(liq / width).astype(np.int64), "long": is_long, "n": notion, "pn": liq * notion})
    g = df.groupby(["bucket", "long"]).agg(n=("n", "sum"), pn=("pn", "sum")).reset_index()
    g = g[g["n"] >= cfg.min_cluster_share * g["n"].max()].nlargest(cfg.max_clusters, "n")
    ts = min(oi.index[-1], ratio.index[-1]).to_pydatetime()           # freshness = the OLDEST input
    clusters = [LiquidationCluster(float(r.pn / r.n), float(r.n), "long" if r.long else "short", ts, MODEL_SOURCE, True)
                for r in g.itertuples()]
    return ModelResult(clusters, "ok", {"added_usd": added, "surviving_usd": float(notion.sum()), "bars": len(d), "coverage": cover})


class PublicModelProvider:
    """Liquidation provider that needs NO API key (uses the same public connector as everything else)."""
    name = "public_model"

    def __init__(self, connector, cfg: PublicModelConfig, candle_limit: int, ratio_kind: RatioKind, maintenance_margin_rate: float):
        self.x, self.cfg, self.limit, self.kind, self.mmr = connector, cfg, candle_limit, ratio_kind, maintenance_margin_rate
        self.minutes = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}[cfg.period]

    async def fetch_clusters(self, symbol: str) -> list[LiquidationCluster]:
        res = await asyncio.gather(self.x.klines(symbol, self.cfg.period, self.limit),
                                   self.x.open_interest_history(symbol, self.cfg.period, 500),
                                   self.x.ratio_history(self.kind, symbol, self.cfg.period, 500), return_exceptions=True)
        bad = [f"{n}: {type(r).__name__}" for n, r in zip(("candles", "open interest", "ratio"), res) if isinstance(r, BaseException)]
        if bad:
            raise RuntimeError("public liquidation model inputs unavailable (" + "; ".join(bad) + ")")
        out = estimate_clusters(*res, self.cfg, self.mmr, self.minutes)
        if not out.clusters:
            raise RuntimeError(out.reason)
        return out.clusters
