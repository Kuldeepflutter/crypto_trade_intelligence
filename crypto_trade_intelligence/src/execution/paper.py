"""Paper broker: simulated fills only. It has NO code path that talks to an exchange."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from config import AppConfig
from src.database.db import Database
from src.execution.guards import pre_trade_check
from src.execution.simulator import PositionSimulator, SimPosition
from src.models import Direction, RatioReading, jsonable
from src.risk.manager import RiskManager
from src.risk.sizing import SizingResult
from src.utils import log_event

log = logging.getLogger("paper")


def _pos_to_json(p: SimPosition) -> str:
    d = asdict(p)
    d["direction"] = p.direction.value
    return json.dumps(d, default=str)


def _pos_from_json(s: str) -> SimPosition:
    d = json.loads(s)
    d["direction"] = Direction(d["direction"])
    d["entry_time"] = datetime.fromisoformat(d["entry_time"])
    d["closed_at"] = datetime.fromisoformat(d["closed_at"]) if d.get("closed_at") else None
    d["exits"] = [tuple(e) for e in d["exits"]]
    return SimPosition(**d)


class PaperBroker:
    mode = "paper"

    def __init__(self, cfg: AppConfig, db: Database, risk: RiskManager):
        self.cfg, self.db, self.risk = cfg, db, risk
        r = cfg.risk
        self.sim = PositionSimulator(r.taker_fee, r.maker_fee, r.max_slippage_pct, r.breakeven_after_tp1,
                                     int(r.max_hold_hours * 12))
        self.positions: dict[int, SimPosition] = {}       # trade_id -> position
        for row in db.open_trades(self.mode):             # rehydrate after restart
            if row.get("payload"):
                self.positions[row["id"]] = _pos_from_json(row["payload"])

    def open_notionals(self) -> list[float]:
        return [p.entry_price * p.remaining for p in self.positions.values()]

    def submit(self, ev, sizing: SizingResult, now: datetime, signal_id: int | None = None,
               fresh_ratio: RatioReading | None = None, market_price: float | None = None) -> tuple[int | None, list[str]]:
        ok, reasons = pre_trade_check(ev, self.cfg.ratio, self.cfg.strategy, now, fresh_ratio)
        if not ok:
            log_event(log, logging.WARNING, "paper_order_rejected", symbol=ev.symbol, reasons=reasons)
            return None, reasons
        if not sizing.ok:
            return None, [sizing.rejected or "sizing failed"]
        if any(p.symbol == ev.symbol for p in self.positions.values()):
            return None, [f"already have an open paper position in {ev.symbol}"]
        decision = self.risk.approve_entry(self.open_notionals(), float(sizing.notional), now)
        if not decision.allowed:
            return None, decision.reasons
        plan, d = ev.plan, ev.direction
        ref = market_price if market_price is not None else plan.entry
        if d.sign * (ref - plan.entry) / plan.entry * 100 > self.cfg.risk.max_slippage_pct:
            return None, [f"price moved {abs(ref - plan.entry) / plan.entry * 100:.3f}% away from the planned entry (> max slippage)"]
        pos = self.sim.open(ev.symbol, d, now, ref, float(sizing.quantity), plan.stop_loss, plan.tps, plan.tp_split,
                            signal_id, ev.setup_kind)
        tps = (plan.tps + [None, None, None])[:3]
        tid = self.db.open_trade(signal_id=signal_id, mode=self.mode, symbol=ev.symbol, direction=d.value,
                                 setup_kind=ev.setup_kind, timeframe="5m", opened_at=now.astimezone(timezone.utc).isoformat(),
                                 entry_price=pos.entry_price, quantity=pos.quantity, stop_loss=plan.stop_loss,
                                 tp1=tps[0], tp2=tps[1], tp3=tps[2], fees=pos.fees, status="open", payload=_pos_to_json(pos))
        self.positions[tid] = pos
        log_event(log, logging.INFO, "paper_position_opened", symbol=ev.symbol, direction=d.value, entry=pos.entry_price,
                  qty=pos.quantity, sl=plan.stop_loss, tps=plan.tps, trade_id=tid)
        return tid, []

    def update(self, symbol: str, candles_5m: pd.DataFrame, funding_rate: float | None = None) -> list[int]:
        """Advance open paper positions in `symbol` using CLOSED 5m candles after their last processed bar."""
        closed: list[int] = []
        for tid, pos in list(self.positions.items()):
            if pos.symbol != symbol:
                continue
            seg = candles_5m.loc[candles_5m["close_time"] > pd.Timestamp(pos.entry_time)]
            for ct, row in seg.iloc[pos.bars_held:].iterrows():       # only bars not yet processed
                if funding_rate is not None and ct.hour % 8 == 0 and ct.minute < 5:
                    self.sim.apply_funding(pos, funding_rate, float(row["open"]), ct.strftime("%Y-%m-%dT%H"))
                if self.sim.on_candle(pos, row["close_time"].to_pydatetime(), float(row["open"]), float(row["high"]),
                                      float(row["low"]), float(row["close"])):
                    break
            self._persist(tid, pos)
            if pos.status == "closed":
                closed.append(tid)
                self.positions.pop(tid)
        return closed

    def _persist(self, tid: int, pos: SimPosition) -> None:
        f = dict(fees=pos.fees, funding=pos.funding, payload=_pos_to_json(pos), mae=pos.mae, mfe=pos.mfe)
        if pos.status == "closed":
            f.update(status="closed", closed_at=pos.closed_at.astimezone(timezone.utc).isoformat(), pnl=pos.pnl,
                     r_multiple=pos.r_multiple, avg_exit_price=pos.avg_exit, exit_reason=pos.exit_reason)
        self.db.update_trade(tid, **f)
