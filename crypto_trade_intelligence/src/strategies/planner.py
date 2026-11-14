"""Entry / stop / take-profit calculator with cost-aware reward-to-risk."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from config import RiskConfig, StrategyConfig
from src.models import Direction
from src.risk.sizing import stop_inside_liquidation
from src.strategies.setups import Trigger
from src.utils import is_finite, round_to_step


@dataclass(frozen=True)
class Target:
    price: float
    kind: str          # "level" | "cluster"
    strength: float    # touches for levels, 0..1 rank for clusters
    label: str


@dataclass
class TradePlan:
    direction: Direction
    entry: float
    entry_zone: tuple[float, float]
    alt_entry: float | None
    stop_loss: float
    tps: list[float]
    tp_labels: list[str]
    tp_split: list[float]
    risk_per_unit: float
    stop_atr: float
    rr_gross: list[float]
    rr_net: float
    costs: dict[str, float]
    nearest_opposing_cluster_pct: float | None
    notes: list[str] = field(default_factory=list)


def _r(x: float, tick: Decimal | None, mode: str) -> float:
    return float(round_to_step(x, tick, mode)) if tick else float(x)


def compute_net_rr(direction: Direction, entry: float, sl: float, tps: list[float], split: list[float], risk: RiskConfig,
                   funding_rate: float | None, hold_hours: float) -> tuple[float, list[float], dict[str, float]]:
    """Weighted reward/risk after taker fees, maker fees on TPs, slippage (entry + stop) and expected funding paid."""
    s = direction.sign
    risk_u = abs(entry - sl)
    slip = risk.max_slippage_pct / 100.0
    funding = max(0.0, s * (funding_rate or 0.0)) * entry * (hold_hours / 8.0)      # only funding PAID is counted
    cost_sl = entry * risk.taker_fee + sl * risk.taker_fee + (entry + sl) * slip + funding
    net_risk = risk_u + cost_sl
    gross = [s * (tp - entry) / risk_u for tp in tps]
    net_rewards = [s * (tp - entry) - (entry * risk.taker_fee + tp * risk.maker_fee + entry * slip + funding) for tp in tps]
    rr = sum(w * r for w, r in zip(split, net_rewards)) / net_risk
    return rr, gross, {"fees_on_stop": entry * risk.taker_fee + sl * risk.taker_fee, "slippage": (entry + sl) * slip, "funding": funding}


def build_plan(direction: Direction, trig: Trigger, atr: float, targets: list[Target], cfg: StrategyConfig,
               risk: RiskConfig, tick: Decimal | None, funding_rate: float | None,
               nearest_cluster_pct: float | None) -> tuple[TradePlan | None, list[str]]:
    s = direction.sign
    notes: list[str] = []
    if not (is_finite(atr) and atr > 0 and is_finite(trig.entry) and is_finite(trig.invalidation)):
        return None, ["invalid ATR / entry / invalidation for plan"]
    entry = _r(trig.entry, tick, "nearest")
    sl = trig.invalidation - s * cfg.stop_atr_buffer * atr
    dist = s * (entry - sl)
    if dist <= 0:
        return None, ["invalidation is on the wrong side of entry"]
    if dist < cfg.min_stop_atr * atr:
        sl, dist = entry - s * cfg.min_stop_atr * atr, cfg.min_stop_atr * atr
        notes.append(f"stop widened to {cfg.min_stop_atr} ATR volatility floor")
    if dist > cfg.max_stop_atr * atr or dist / entry * 100 > cfg.max_stop_pct:
        return None, [f"entry too far from invalidation (stop {dist / atr:.2f} ATR / {dist / entry * 100:.2f}% > limit)"]
    sl = _r(sl, tick, "down" if s > 0 else "up")
    risk_u = abs(entry - sl)
    if risk_u <= 0:
        return None, ["zero risk distance after rounding"]
    if not stop_inside_liquidation(direction, entry, sl, risk.leverage, risk.maintenance_margin_rate, risk.liquidation_safety):
        return None, [f"stop is too close to the estimated liquidation price at {risk.leverage:g}x leverage; lower leverage"]

    # targets strictly beyond entry, nearest first; targets are placed just in front of the level
    cands = sorted((t for t in targets if s * (t.price - entry) > 0), key=lambda t: s * (t.price - entry))
    obstacle = next((t for t in cands if (t.kind == "cluster" and t.strength >= 0.2) or (t.kind == "level" and t.strength >= cfg.obstacle_min_touches)), None)
    if obstacle is not None and s * (obstacle.price - entry) < cfg.min_tp1_r * risk_u:
        return None, [f"opposing {obstacle.label} blocks the path ({s * (obstacle.price - entry) / risk_u:.2f}R < {cfg.min_tp1_r}R)"]
    tps: list[float] = []
    labels: list[str] = []
    for t in cands:
        px = _r(t.price - s * cfg.tp_buffer_atr * atr, tick, "down" if s > 0 else "up")
        if s * (px - entry) < cfg.min_tp1_r * risk_u:
            continue
        if tps and s * (px - tps[-1]) < 0.5 * risk_u:
            continue
        tps.append(px)
        labels.append(t.label)
        if len(tps) == 3:
            break
    if not tps:
        return None, ["no logical take-profit level (no structure/liquidity target at >= min R)"]
    split = [x / sum(cfg.tp_split[:len(tps)]) for x in cfg.tp_split[:len(tps)]]

    rr_net, gross, costs = compute_net_rr(direction, entry, sl, tps, split, risk, funding_rate, cfg.expected_hold_hours)
    if rr_net < cfg.min_rr:
        return None, [f"net reward/risk {rr_net:.2f} < minimum {cfg.min_rr} after fees, slippage and funding"]

    zone = (entry - s * 0.25 * atr, entry) if s > 0 else (entry, entry + 0.25 * atr)
    alt = trig.confirm_extreme + s * 0.02 * atr
    alt_entry = _r(alt, tick, "nearest") if s * (alt - entry) > 0 else None
    return TradePlan(direction, entry, zone, alt_entry, sl, tps, labels, split, risk_u, dist / atr, gross, rr_net,
                     costs, nearest_cluster_pct, notes), []
