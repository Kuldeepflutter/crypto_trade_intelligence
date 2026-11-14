"""Portfolio-level risk limits and the daily-loss kill switch."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

from config import RiskConfig


@dataclass
class RiskDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, cfg: RiskConfig, daily_pnl_fn: Callable[[datetime], float] | None = None,
                 total_pnl_fn: Callable[[], float] | None = None):
        self.cfg = cfg
        self._daily = daily_pnl_fn or (lambda now: 0.0)
        self._total = total_pnl_fn or (lambda: 0.0)

    @property
    def equity(self) -> float:
        return self.cfg.account_equity + self._total()

    @property
    def daily_loss_limit(self) -> float:
        return self.cfg.account_equity * self.cfg.max_daily_loss_pct / 100.0

    def daily_pnl(self, now: datetime) -> float:
        return self._daily(now)

    def kill_switch_active(self, now: datetime) -> tuple[bool, str]:
        pnl = self.daily_pnl(now)
        if pnl <= -self.daily_loss_limit:
            return True, f"KILL SWITCH: daily realized PnL {pnl:.2f} <= -{self.daily_loss_limit:.2f}; no new entries until the next UTC day"
        return False, ""

    def remaining_exposure(self, open_notionals: Iterable[float]) -> float:
        cap = self.equity * self.cfg.max_total_exposure_pct / 100.0
        return cap - sum(open_notionals)

    def approve_entry(self, open_notionals: list[float], new_notional: float, now: datetime | None = None) -> RiskDecision:
        now = now or datetime.now(timezone.utc)
        reasons: list[str] = []
        killed, why = self.kill_switch_active(now)
        if killed:
            reasons.append(why)
        if len(open_notionals) >= self.cfg.max_positions:
            reasons.append(f"max simultaneous positions reached ({self.cfg.max_positions})")
        if new_notional > self.remaining_exposure(open_notionals) + 1e-9:
            reasons.append("total exposure limit would be exceeded")
        return RiskDecision(not reasons, reasons)
