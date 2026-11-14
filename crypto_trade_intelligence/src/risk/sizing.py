"""Risk-budget position sizing. Leverage only affects margin, never the risk budget."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from src.models import Direction, SymbolRules
from src.utils import is_finite, round_to_step, to_decimal


@dataclass
class SizingResult:
    quantity: Decimal = Decimal(0)
    notional: Decimal = Decimal(0)
    margin: Decimal = Decimal(0)
    risk_budget: float = 0.0
    risk_actual: float = 0.0
    rejected: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.rejected is None and self.quantity > 0


def liquidation_price(direction: Direction, entry: float, leverage: float, mmr: float) -> float:
    """Approximate isolated-margin liquidation price (ignores fees/funding; exchange tiers differ)."""
    return entry * (1 - 1 / leverage + mmr) if direction is Direction.LONG else entry * (1 + 1 / leverage - mmr)


def stop_inside_liquidation(direction: Direction, entry: float, stop: float, leverage: float, mmr: float, safety: float) -> bool:
    liq = liquidation_price(direction, entry, leverage, mmr)
    return abs(entry - stop) <= safety * abs(entry - liq)


def cost_per_unit(entry: float, exit_price: float, fee_rate: float, slippage_pct: float) -> float:
    return (entry + exit_price) * (fee_rate + slippage_pct / 100.0)


def size_position(entry: float, stop: float, equity: float, risk_pct: float, leverage: float,
                  rules: SymbolRules | None, margin_alloc_pct: float, fee_rate: float, slippage_pct: float,
                  remaining_exposure_notional: float | None = None) -> SizingResult:
    res = SizingResult()
    if rules is None:
        res.rejected = "exchange trading rules unavailable (tick/step/min size) - will not guess"
        return res
    if not all(is_finite(x) for x in (entry, stop, equity, risk_pct, leverage)) or entry <= 0 or stop <= 0:
        res.rejected = "invalid sizing inputs"
        return res
    res.risk_budget = equity * risk_pct / 100.0
    unit_loss = abs(entry - stop) + cost_per_unit(entry, stop, fee_rate, slippage_pct)
    if abs(entry - stop) <= 0 or unit_loss <= 0:
        res.rejected = "zero stop distance"
        return res
    qty = res.risk_budget / unit_loss
    caps = {"risk": qty, "margin": equity * margin_alloc_pct / 100.0 * leverage / entry}
    if remaining_exposure_notional is not None:
        caps["exposure"] = max(0.0, remaining_exposure_notional) / entry
    binding = min(caps, key=caps.get)
    qty = caps[binding]
    if rules.max_qty:
        qty = min(qty, float(rules.max_qty))
    q = round_to_step(qty, rules.step_size, "down")
    notional = q * to_decimal(entry)
    if q < rules.min_qty or notional < rules.min_notional:
        # Try the smallest legal size, but only if it still respects every cap (never exceed the risk budget).
        q_min = max(round_to_step(rules.min_qty, rules.step_size, "up"),
                    round_to_step(rules.min_notional / to_decimal(entry), rules.step_size, "up"))
        if float(q_min) <= min(caps.values()) + 1e-12:
            q, notional = q_min, q_min * to_decimal(entry)
            res.notes.append("raised to exchange minimum size (still within every cap)")
        else:
            res.rejected = (f"risk budget {res.risk_budget:.2f} USDT cannot fund the exchange minimum size "
                            f"(min qty {rules.min_qty}, min notional {rules.min_notional}); binding cap: {binding}")
            return res
    res.quantity, res.notional = q, notional
    res.margin = notional / to_decimal(leverage)
    res.risk_actual = float(q) * unit_loss
    if res.risk_actual > res.risk_budget * (1 + 1e-9):
        res.rejected = "internal check failed: sized risk exceeds budget"
        res.quantity = Decimal(0)
        return res
    res.notes.append(f"binding constraint: {binding}")
    return res
