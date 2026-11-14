"""Pre-trade guard used by alerts, paper trading and live execution.

It re-validates the mandatory SHORT filter at the moment of use (so a signal that was fine two
minutes ago but whose ratio is now stale or has dropped below 1.0 cannot be alerted or ordered).
"""
from __future__ import annotations

import math
from datetime import datetime

from config import RatioConfig, StrategyConfig
from src.models import Direction, Outcome, RatioKind, RatioReading
from src.positioning.ratio import ratio_gate


def pre_trade_check(ev, ratio_cfg: RatioConfig, strat_cfg: StrategyConfig, now: datetime,
                    fresh_ratio: RatioReading | None = None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if ev.outcome is not Outcome.TRADE_SETUP or ev.direction is None or ev.plan is None:
        return False, ["not a qualified TRADE SETUP"]
    plan = ev.plan
    if ev.direction is Direction.SHORT:
        reading = fresh_ratio if fresh_ratio is not None else ev.ratio
        g = ratio_gate(reading, RatioKind(ratio_cfg.kind), now, ratio_cfg.max_age_seconds)
        if not g.passed:
            reasons.append(g.reason)
    s = ev.direction.sign
    vals = [plan.entry, plan.stop_loss, *plan.tps]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        reasons.append("plan contains non-finite prices")
    else:
        if s * (plan.entry - plan.stop_loss) <= 0:
            reasons.append("stop-loss is on the wrong side of entry")
        if any(s * (tp - plan.entry) <= 0 for tp in plan.tps):
            reasons.append("take-profit on the wrong side of entry")
    if plan.rr_net < strat_cfg.min_rr:
        reasons.append(f"net reward/risk {plan.rr_net:.2f} below minimum {strat_cfg.min_rr}")
    return not reasons, reasons
