"""Event-driven backtester over 5m candles.

* Higher timeframes are resampled from the 5m series and only COMPLETE bars (ending <= decision time)
  are visible -> no look-ahead.
* A decision is made at the CLOSE of a 5m bar; the order fills at the NEXT bar's open (+ slippage).
* Fees, slippage and funding are charged; stop-first intrabar convention (see execution.simulator).
* Historical liquidation heatmaps and order books are normally unavailable, so those features are
  switched off and the result is labelled a REDUCED-FEATURE backtest. SHORTs additionally require a
  historical long/short ratio series; without one the mandatory filter blocks every SHORT.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import pandas as pd

from config import AppConfig, ROOT
from src.backtesting.metrics import baseline_buy_hold, baseline_ema_cross, by_group, compute_metrics
from src.execution.simulator import PositionSimulator
from src.indicators.core import resample_ohlcv
from src.models import Direction, Outcome, RatioKind, RatioReading, SymbolRules
from src.positioning.oi import pct_change_over
from src.positioning.ratio import ratio_changes
from src.risk.sizing import size_position
from src.strategies.engine import StrategyEngine, StrategyInput
from src.strategies.planner import compute_net_rr

DEFAULT_RULES = SymbolRules("BACKTEST", Decimal("0.01"), Decimal("0.001"), Decimal("0.001"), Decimal("5"), None, "assumed defaults")
RULES = {"15m": "15min", "1h": "1h", "4h": "4h"}


@dataclass
class BacktestResult:
    symbol: str
    metrics: dict
    trades: list[dict]
    rejected_signals: list[dict]
    rejection_counts: dict[str, int]
    reduced_features: list[str]
    baselines: list[dict]
    by_regime: dict
    start: str = ""
    end: str = ""
    steps: int = 0
    params: dict = field(default_factory=dict)


class Backtester:
    def __init__(self, cfg: AppConfig, df5: pd.DataFrame, symbol: str = "BTCUSDT", ratio: pd.DataFrame | None = None,
                 oi: pd.DataFrame | None = None, funding: pd.DataFrame | None = None, rules: SymbolRules | None = None,
                 ratio_kind: RatioKind = RatioKind.GLOBAL_ACCOUNT):
        self.cfg = cfg.model_copy(deep=True)
        self.cfg.strategy.liquidation_mode = "optional"      # no historical heatmap -> reduced-feature test
        self.cfg.strategy.require_orderbook = False          # no historical order book
        self.cfg.strategy.min_volume_24h_usd = 0.0
        self.df5, self.symbol, self.ratio, self.oi, self.funding = df5, symbol, ratio, oi, funding
        self.rules = rules or DEFAULT_RULES
        self.ratio_kind = ratio_kind
        self.engine = StrategyEngine(self.cfg.strategy, self.cfg.risk, self.cfg.ratio)
        self._tf = {}
        for tf, rule in RULES.items():
            r = resample_ohlcv(df5, rule)
            span = pd.Timedelta(rule)
            r["bar_end"] = r.index + span - pd.Timedelta(milliseconds=1)
            self._tf[tf] = (r, r["bar_end"].to_numpy())
        self._ct = df5["close_time"].to_numpy()

    @property
    def warmup_bars(self) -> int:
        return (self.cfg.strategy.ema_slow + 12) * 48         # 4h EMA200 needs ~212 four-hour bars

    def reduced_features(self) -> list[str]:
        out = ["liquidation heatmap (no historical data)", "order-book depth/imbalance (no historical data)"]
        if self.ratio is None:
            out.append("long/short ratio (none supplied -> mandatory filter blocks ALL shorts)")
        if self.oi is None:
            out.append("open interest")
        if self.funding is None:
            out.append("funding (not charged / not in R/R)")
        out.append("exchange rules: " + self.rules.source)
        return out

    def _windows(self, i: int) -> dict[str, pd.DataFrame]:
        lim = self.cfg.strategy.candle_limit
        t = self._ct[i]
        out = {"5m": self.df5.iloc[max(0, i - lim + 1): i + 1]}
        for tf, (frame, ends) in self._tf.items():
            k = int(np.searchsorted(ends, t, side="right"))
            out[tf] = frame.iloc[max(0, k - lim):k].drop(columns="bar_end")
        return out

    def _input(self, i: int, candles: dict) -> StrategyInput:
        now = self.df5["close_time"].iloc[i].to_pydatetime()
        ts = self.df5["close_time"].iloc[i]
        reading, rch = None, {}
        if self.ratio is not None:
            h = self.ratio.loc[:ts]
            if len(h):
                reading = RatioReading(float(h["ratio"].iloc[-1]), self.ratio_kind, h.index[-1].to_pydatetime(), "backtest-history", self.cfg.ratio.period)
                rch = ratio_changes(h.tail(30), now)
        oi_chg = price_chg = None
        if self.oi is not None:
            h = self.oi.loc[:ts]
            oi_chg = pct_change_over(h.tail(60), "oi", self.cfg.strategy.oi_window_minutes)
            if oi_chg is not None:
                c = candles["5m"]
                w = self.cfg.strategy.oi_window_minutes // 5
                if len(c) > w:
                    price_chg = (float(c["close"].iloc[-1]) / float(c["close"].iloc[-1 - w]) - 1) * 100
        fr = None
        if self.funding is not None:
            f = self.funding.loc[:ts]
            fr = float(f["rate"].iloc[-1]) if len(f) else None
        return StrategyInput(self.symbol, now, float(candles["5m"]["close"].iloc[-1]), candles, reading, rch, oi_chg,
                             price_chg, fr, None, None, None, None, self.rules.tick_size, [])

    def run(self, start: int | None = None, end: int | None = None, step_bars: int = 3) -> BacktestResult:
        cfg, r = self.cfg, self.cfg.risk
        start = max(start or 0, self.warmup_bars)
        end = min(end or len(self.df5) - 2, len(self.df5) - 2)
        sim = PositionSimulator(r.taker_fee, r.maker_fee, r.max_slippage_pct, r.breakeven_after_tp1, int(r.max_hold_hours * 12))
        trades, rejected, reasons = [], [], Counter()
        equity, pos, pending_ctx = r.account_equity, None, None
        day_pnl: dict[str, float] = {}
        steps = 0
        strat_log = logging.getLogger("strategy")
        old = strat_log.level
        strat_log.setLevel(logging.WARNING)                  # per-step decision logs would flood a long backtest
        try:
            for i in range(start, end, 1):
                row = self.df5.iloc[i]
                if pos is not None:                           # manage the open position on this closed bar
                    f = None
                    if self.funding is not None and row["close_time"].hour % 8 == 7 and row["close_time"].minute >= 55:
                        h = self.funding.loc[:row["close_time"]]
                        f = float(h["rate"].iloc[-1]) if len(h) else None
                    if f is not None:
                        sim.apply_funding(pos, f, float(row["open"]), row["close_time"].strftime("%Y-%m-%dT%H"))
                    if sim.on_candle(pos, row["close_time"], float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])):
                        pnl = pos.pnl
                        equity += pnl
                        day = row["close_time"].strftime("%Y-%m-%d")
                        day_pnl[day] = day_pnl.get(day, 0.0) + pnl
                        trades.append({"entry_time": str(pos.entry_time), "exit_time": str(pos.closed_at), "direction": pos.direction.value,
                                       "setup_kind": pos.setup_kind, "entry": pos.entry_price, "avg_exit": pos.avg_exit,
                                       "qty": pos.quantity, "pnl": pnl, "r_multiple": pos.r_multiple, "fees": pos.fees,
                                       "funding": pos.funding, "mae": pos.mae, "mfe": pos.mfe, "exit_reason": pos.exit_reason,
                                       "bars": pos.bars_held, **(pending_ctx or {})})
                        pos = None
                    continue
                if (i - start) % step_bars != 0:
                    continue
                steps += 1
                candles = self._windows(i)
                inp = self._input(i, candles)
                ev = self.engine.evaluate(inp)
                if ev.outcome is Outcome.TRADE_SETUP:
                    reasons["accepted setup"] += 1
                for name, cand in ev.candidates.items():
                    if cand.rejections:
                        reasons[re.sub(r"\d+(\.\d+)?", "#", f"{name}: {cand.rejections[0]}")[:100]] += 1
                if not ev.candidates:
                    reasons[(ev.rejections or ["no evaluation"])[0][:90]] += 1
                if ev.outcome is not Outcome.TRADE_SETUP:
                    if ev.trigger is not None or ev.outcome is Outcome.WATCH:
                        rejected.append({"time": str(inp.now), "outcome": ev.outcome.value, "direction": ev.direction.value if ev.direction else None,
                                         "reasons": ev.rejections[:3]})
                    continue
                day = inp.now.strftime("%Y-%m-%d")
                if day_pnl.get(day, 0.0) <= -r.account_equity * r.max_daily_loss_pct / 100:
                    reasons["kill switch (daily loss)"] += 1
                    continue
                nxt = self.df5.iloc[i + 1]                    # fill at the NEXT bar's open
                d, plan = ev.direction, ev.plan
                fill = sim.entry_fill(d, float(nxt["open"]))
                if d.sign * (fill - plan.stop_loss) <= 0 or any(d.sign * (tp - fill) <= 0 for tp in plan.tps):
                    reasons["gap invalidated plan"] += 1
                    continue
                rr, _, _ = compute_net_rr(d, fill, plan.stop_loss, plan.tps, plan.tp_split, r, ev.funding_rate, cfg.strategy.expected_hold_hours)
                if rr < cfg.strategy.min_rr:
                    reasons["R/R invalid at actual fill"] += 1
                    continue
                sz = size_position(fill, plan.stop_loss, equity, r.risk_per_trade_pct, r.leverage, self.rules,
                                   r.margin_allocation_pct, r.taker_fee, r.max_slippage_pct, r.account_equity * r.max_total_exposure_pct / 100)
                if not sz.ok:
                    reasons["sizing: " + str(sz.rejected)[:60]] += 1
                    continue
                pos = sim.open(self.symbol, d, nxt["close_time"] - pd.Timedelta(minutes=5), float(nxt["open"]), float(sz.quantity),
                               plan.stop_loss, plan.tps, plan.tp_split, None, ev.setup_kind)
                pending_ctx = {"regime_4h": ev.trend_labels.get("4h"), "bias": ev.bias, "score": ev.score, "rr_planned": plan.rr_net}
                # NOTE: the entry bar itself is processed on the next loop iteration (i+1), i.e. after the fill at its open.
        finally:
            strat_log.setLevel(old)
        seg = self.df5.iloc[start:end]
        fee, slip = r.taker_fee, r.max_slippage_pct
        return BacktestResult(
            self.symbol, compute_metrics(trades, r.account_equity), trades, rejected, dict(reasons.most_common(15)),
            self.reduced_features(), [baseline_buy_hold(seg, fee, slip), baseline_ema_cross(seg, fee, slip, allow_short=True),
                                      baseline_ema_cross(seg, fee, slip, allow_short=False)],
            by_group(trades, "regime_4h", r.account_equity), str(seg.index[0]), str(seg.index[-1]), steps)
