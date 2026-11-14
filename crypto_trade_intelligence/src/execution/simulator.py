"""Candle-based position simulator shared by paper trading and the backtester.

Conservative conventions: if the stop and a take-profit are both touched inside one candle the STOP
fills first; gaps through the stop fill at the (worse) open; stops/entries/timeouts pay taker fees
and slippage; take-profits are limit orders (maker fee, no slippage) and must be traded THROUGH.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.models import Direction


@dataclass
class SimPosition:
    symbol: str
    direction: Direction
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    tps: list[float]
    tp_split: list[float]
    signal_id: int | None = None
    setup_kind: str | None = None
    remaining: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    gross_pnl: float = 0.0
    exits: list[tuple[str, float, float, str]] = field(default_factory=list)   # (time, price, qty, reason)
    tp_hit: int = 0
    worst_price: float = 0.0
    best_price: float = 0.0
    bars_held: int = 0
    status: str = "open"
    exit_reason: str | None = None
    closed_at: datetime | None = None
    sl_moved_to_be: bool = False
    initial_stop: float = 0.0
    last_funding_slot: str | None = None

    def __post_init__(self):
        self.remaining = self.remaining or self.quantity
        self.worst_price = self.worst_price or self.entry_price
        self.best_price = self.best_price or self.entry_price
        self.initial_stop = self.initial_stop or self.stop_loss

    @property
    def sign(self) -> int:
        return self.direction.sign

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.initial_stop) * self.quantity

    @property
    def pnl(self) -> float:
        return self.gross_pnl - self.fees - self.funding

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.risk_amount if self.risk_amount > 0 else 0.0

    @property
    def mae(self) -> float:      # max adverse excursion in price units
        return max(0.0, self.sign * (self.entry_price - self.worst_price))

    @property
    def mfe(self) -> float:
        return max(0.0, self.sign * (self.best_price - self.entry_price))

    @property
    def avg_exit(self) -> float | None:
        q = sum(e[2] for e in self.exits)
        return sum(e[1] * e[2] for e in self.exits) / q if q else None


class PositionSimulator:
    def __init__(self, taker_fee: float, maker_fee: float, slippage_pct: float,
                 breakeven_after_tp1: bool = False, max_hold_bars: int | None = None):
        self.taker, self.maker, self.slip = taker_fee, maker_fee, slippage_pct / 100.0
        self.be, self.max_hold = breakeven_after_tp1, max_hold_bars

    def entry_fill(self, direction: Direction, ref_price: float) -> float:
        return ref_price * (1 + direction.sign * self.slip)          # adverse slippage

    def open(self, symbol: str, direction: Direction, time: datetime, ref_price: float, qty: float, sl: float,
             tps: list[float], split: list[float], signal_id=None, setup_kind=None) -> SimPosition:
        fill = self.entry_fill(direction, ref_price)
        pos = SimPosition(symbol, direction, time, fill, qty, sl, list(tps), list(split), signal_id, setup_kind)
        pos.fees += fill * qty * self.taker
        return pos

    def _exit(self, pos: SimPosition, qty: float, price: float, time, reason: str, maker: bool) -> None:
        qty = min(qty, pos.remaining)
        pos.gross_pnl += pos.sign * (price - pos.entry_price) * qty
        pos.fees += price * qty * (self.maker if maker else self.taker)
        pos.remaining -= qty
        pos.exits.append((str(time), price, qty, reason))
        if pos.remaining <= pos.quantity * 1e-9:
            pos.remaining, pos.status, pos.exit_reason, pos.closed_at = 0.0, "closed", reason, time

    def on_candle(self, pos: SimPosition, time, o: float, h: float, l: float, c: float) -> bool:
        """Advance one CLOSED candle. `time` is the candle close time. Returns True when the position is closed."""
        if pos.status != "open":
            return True
        s = pos.sign
        pos.bars_held += 1
        pos.worst_price = min(pos.worst_price, l) if s > 0 else max(pos.worst_price, h)
        pos.best_price = max(pos.best_price, h) if s > 0 else min(pos.best_price, l)
        sl = pos.stop_loss
        stop_hit_open = s * (o - sl) <= 0
        stop_hit = stop_hit_open or (l <= sl if s > 0 else h >= sl)
        if stop_hit:
            raw = o if stop_hit_open else sl                       # gap -> fill at the worse open
            fill = raw * (1 - s * self.slip)
            self._exit(pos, pos.remaining, fill, time, "breakeven_stop" if pos.sl_moved_to_be else "stop_loss", maker=False)
            return True
        while pos.tp_hit < len(pos.tps) and pos.status == "open":
            tp = pos.tps[pos.tp_hit]
            if not ((h > tp) if s > 0 else (l < tp)):
                break
            last = pos.tp_hit == len(pos.tps) - 1
            qty = pos.remaining if last else pos.quantity * pos.tp_split[pos.tp_hit]
            self._exit(pos, qty, tp, time, f"tp{pos.tp_hit + 1}", maker=True)
            pos.tp_hit += 1
            if self.be and pos.tp_hit == 1 and pos.status == "open":
                pos.stop_loss, pos.sl_moved_to_be = pos.entry_price, True
        if pos.status == "open" and self.max_hold and pos.bars_held >= self.max_hold:
            self._exit(pos, pos.remaining, c * (1 - s * self.slip), time, "timeout", maker=False)
        return pos.status != "open"

    def apply_funding(self, pos: SimPosition, rate: float, mark_price: float, slot: str) -> None:
        """Charge funding once per slot (e.g. '2025-01-01T08'); longs pay when rate > 0."""
        if pos.status != "open" or pos.last_funding_slot == slot:
            return
        pos.last_funding_slot = slot
        pos.funding += pos.sign * rate * mark_price * pos.remaining
