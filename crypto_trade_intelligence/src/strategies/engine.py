"""Strategy engine: hard filters -> triggers -> confirmations -> plan -> score -> outcome.

Every candidate (LONG and SHORT) is evaluated independently and every acceptance/rejection reason
is logged and stored. A rejected SHORT never makes a LONG acceptable.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import pandas as pd

from config import RatioConfig, RiskConfig, StrategyConfig
from src.indicators.core import validate_candles
from src.liquidation.heatmap import HeatmapContext, heatmap_support
from src.market_structure.swings import find_levels, merge_levels
from src.market_structure.trend import MTFTrend, analyze_mtf, reversal_confirmed
from src.models import Direction, Outcome, OrderBookMetrics, RatioKind, RatioReading, TrendLabel, jsonable
from src.positioning.oi import oi_contradiction
from src.positioning.ratio import ratio_gate
from src.strategies.planner import Target, TradePlan, build_plan
from src.strategies.scoring import aggregate, compute_components
from src.strategies.setups import Trigger, detect_triggers
from src.utils import TF_SECONDS, is_finite, log_event

log = logging.getLogger("strategy")


@dataclass
class StrategyInput:
    symbol: str
    now: datetime
    price: float | None
    candles: dict[str, pd.DataFrame]
    ratio: RatioReading | None = None
    ratio_changes: dict[str, float | None] = field(default_factory=dict)
    oi_change_pct: float | None = None
    price_change_pct: float | None = None          # over the same window as oi_change_pct
    funding_rate: float | None = None
    volume_change_pct: float | None = None
    orderbook: OrderBookMetrics | None = None
    heatmap: HeatmapContext | None = None
    volume_24h_usd: float | None = None
    tick_size: Decimal | None = None
    reduced_features: list[str] = field(default_factory=list)


@dataclass
class CandidateResult:
    direction: Direction
    outcome: Outcome = Outcome.NO_TRADE
    score: float | None = None
    components: dict = field(default_factory=dict)
    passed: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    trigger: Trigger | None = None
    plan: TradePlan | None = None


@dataclass
class Evaluation:
    symbol: str
    timestamp: datetime
    price: float | None
    outcome: Outcome = Outcome.NO_TRADE
    direction: Direction | None = None
    score: float | None = None
    components: dict = field(default_factory=dict)
    passed: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    trigger: Trigger | None = None
    plan: TradePlan | None = None
    trend_labels: dict[str, str] = field(default_factory=dict)
    bias: str = "unknown"
    ratio: RatioReading | None = None
    ratio_kind: str = ""
    oi_change_pct: float | None = None
    funding_rate: float | None = None
    volume_24h_usd: float | None = None
    cluster_above: dict | None = None
    cluster_below: dict | None = None
    reduced_features: list[str] = field(default_factory=list)
    atr15: float | None = None
    candidates: dict[str, CandidateResult] = field(default_factory=dict)
    other_notes: list[str] = field(default_factory=list)   # why the NON-selected direction was rejected

    @property
    def setup_kind(self) -> str | None:
        return self.trigger.kind if self.trigger else None

    def to_dict(self) -> dict:
        d = jsonable(self)
        return d


def _cluster_dict(rc) -> dict | None:
    return None if rc is None else {"price": rc.cluster.price, "distance_pct": rc.distance_pct,
                                    "notional_usd": rc.cluster.notional_usd, "estimated": True}


class StrategyEngine:
    def __init__(self, cfg: StrategyConfig, risk: RiskConfig, ratio_cfg: RatioConfig):
        self.cfg, self.risk, self.ratio_cfg = cfg, risk, ratio_cfg

    # ------------------------------------------------------------ validation
    def _validate(self, inp: StrategyInput) -> list[str]:
        cfg, out = self.cfg, []
        if inp.now.tzinfo is None:
            return ["internal: naive timestamp"]
        if not is_finite(inp.price) or inp.price <= 0:
            out.append("current price unavailable or invalid (no price is ever fabricated)")
        for tf in ("5m", "15m", "1h", "4h"):
            df = inp.candles.get(tf)
            problem = validate_candles(df, cfg.ema_slow + 10)
            if problem:
                out.append(f"{tf} candles: {problem}")
                continue
            age = (inp.now - df["close_time"].iloc[-1]).total_seconds()
            if age > (cfg.max_candle_lag_bars + 1) * TF_SECONDS[tf]:
                out.append(f"{tf} candles stale (last close {age:.0f}s ago)")
        if not out and abs(inp.price / float(inp.candles["5m"]["close"].iloc[-1]) - 1) > 0.10:
            out.append("price deviates >10% from last 5m close: inconsistent/stale data")
        if inp.volume_24h_usd is not None and inp.volume_24h_usd < cfg.min_volume_24h_usd:
            out.append(f"24h volume {inp.volume_24h_usd:,.0f} USDT below minimum {cfg.min_volume_24h_usd:,.0f}")
        return out

    # ------------------------------------------------------------ main entry
    def evaluate(self, inp: StrategyInput) -> Evaluation:
        ev = Evaluation(inp.symbol, inp.now, inp.price, ratio=inp.ratio, ratio_kind=self.ratio_cfg.kind,
                        oi_change_pct=inp.oi_change_pct, funding_rate=inp.funding_rate,
                        volume_24h_usd=inp.volume_24h_usd, reduced_features=list(inp.reduced_features))
        problems = self._validate(inp)
        if problems:
            ev.rejections = problems
            return self._finish(ev)
        mtf = analyze_mtf(inp.candles, self.cfg)
        ev.trend_labels, ev.bias = mtf.labels(), mtf.bias
        ev.atr15 = mtf.frames["15m"].atr
        if inp.heatmap and inp.heatmap.available:
            ev.cluster_above, ev.cluster_below = _cluster_dict(inp.heatmap.nearest_above), _cluster_dict(inp.heatmap.nearest_below)
        if mtf.bias == "unknown" or not is_finite(ev.atr15):
            ev.rejections = ["trend data insufficient: " + "; ".join(mtf.notes or ["indicators unavailable"])]
            return self._finish(ev)
        targets = self._targets(inp, mtf)
        for d in Direction:
            ev.candidates[d.value] = self._candidate(d, inp, mtf, targets, ev)
        order = {Outcome.TRADE_SETUP: 0, Outcome.WATCH: 1, Outcome.NO_TRADE: 2}
        best = sorted(ev.candidates.values(), key=lambda c: (order[c.outcome], -(c.score or 0.0)))[0]
        ev.outcome, ev.score, ev.components = best.outcome, best.score, best.components
        ev.passed, ev.plan, ev.trigger = best.passed, best.plan, best.trigger
        ev.direction = best.direction if (best.outcome is Outcome.TRADE_SETUP or best.trigger is not None) else None
        ev.rejections = list(best.rejections)
        other = [c for c in ev.candidates.values() if c is not best]
        notes = [f"{c.direction.value}: {r}" for c in other for r in c.rejections]
        if best.outcome is Outcome.TRADE_SETUP:
            ev.other_notes = notes                       # e.g. "SHORT: SHORT blocked: ... ratio 0.850 < 1.0"
        else:
            ev.rejections = [f"{best.direction.value}: {r}" for r in best.rejections] + notes
        if mtf.bias in ("conflict", "neutral") and best.outcome is not Outcome.TRADE_SETUP:
            ev.rejections = list(dict.fromkeys(mtf.notes + ev.rejections))
        return self._finish(ev)

    def _finish(self, ev: Evaluation) -> Evaluation:
        log_event(log, logging.INFO, "setup_decision", symbol=ev.symbol, outcome=ev.outcome.value,
                  direction=ev.direction.value if ev.direction else None, score=ev.score, rejections=ev.rejections,
                  passed=ev.passed, bias=ev.bias, reduced_features=ev.reduced_features,
                  candidates={k: {"outcome": c.outcome.value, "score": c.score, "rejections": c.rejections, "passed": c.passed}
                              for k, c in ev.candidates.items()})
        return ev

    # ------------------------------------------------------------ helpers
    def _targets(self, inp: StrategyInput, mtf: MTFTrend) -> list[Target]:
        cfg, atr = self.cfg, mtf.frames["15m"].atr
        tol = 0.5 * atr
        lv = merge_levels(find_levels(inp.candles["15m"], tol, cfg.swing_left, cfg.swing_right) +
                          find_levels(inp.candles["1h"], tol, cfg.swing_left, cfg.swing_right) +
                          find_levels(inp.candles["4h"], tol * 2, cfg.swing_left, cfg.swing_right), tol)
        out = [Target(l.price, "level", float(l.touches), f"level {l.price:.6g} ({l.touches} touches)") for l in lv]
        if inp.heatmap and inp.heatmap.available:
            out += [Target(c.cluster.price, "cluster", c.rank, f"estimated liquidation cluster {c.cluster.price:.6g}")
                    for c in inp.heatmap.clusters if c.swept is False and c.rank > 0]
        return out

    def _htf(self, d: Direction, mtf: MTFTrend) -> tuple[bool, float, str]:
        want = "bullish" if d is Direction.LONG else "bearish"
        if mtf.bias == want:
            return True, mtf.bias_quality, f"higher-timeframe bias {want}"
        ok, why = reversal_confirmed(d.sign, mtf, self.cfg)
        if ok:
            return True, 0.5, f"confirmed {want} reversal ({why})"
        return False, 0.0, f"higher-timeframe bias is {mtf.bias}; {d.value} needs {want} bias or a confirmed reversal"

    def _structure_ok(self, d: Direction, mtf: MTFTrend, kind: str) -> tuple[bool, str]:
        m15, m5, h1 = mtf.frames["15m"], mtf.frames["5m"], mtf.frames["1h"]
        want, against = ("bullish", "bearish") if d is Direction.LONG else ("bearish", "bullish")
        ok15 = (m15.struct_dir == d.sign or m15.structure.choch == want or m15.structure.bos == want
                or m5.structure.choch == want or m5.structure.bos == want
                or (kind == "liquidity_sweep" and h1.struct_dir == d.sign))   # reclaimed sweep inside an intact 1h structure
        bad5 = kind != "liquidity_sweep" and (m5.structure.bos == against or m5.structure.choch == against)
        if not ok15:
            return False, "no 15m/5m structure confirmation of the direction (no trend, BOS or CHoCH in its favour)"
        if bad5:
            return False, "recent 5m structure break against the trade"
        return True, "15m structure confirms"

    # ------------------------------------------------------------ per direction
    def _candidate(self, d: Direction, inp: StrategyInput, mtf: MTFTrend, targets: list[Target], ev: Evaluation) -> CandidateResult:
        cfg = self.cfg
        c = CandidateResult(d)
        # 1) mandatory SHORT filter - enforced here, before any SHORT can be generated
        if d is Direction.SHORT:
            g = ratio_gate(inp.ratio, RatioKind(self.ratio_cfg.kind), inp.now, self.ratio_cfg.max_age_seconds)
            if not g.passed:
                c.rejections.append(g.reason)
                return c
            c.passed.append(g.reason)
        # 2) higher timeframe
        htf_ok, quality, why = self._htf(d, mtf)
        if mtf.bias == "conflict" and not htf_ok:
            c.outcome = Outcome.WATCH
            c.rejections.append("higher timeframes conflict -> WAIT (no forced trade)")
            return c
        if not htf_ok:
            c.rejections.append(why)
            return c
        c.passed.append(why)
        # 3) trigger from price action (a liquidation cluster alone can never trigger)
        trigs = detect_triggers(d, inp.candles, ev.atr15, cfg)
        valid = [t for t in trigs if t.status == "valid"]
        if not valid:
            if trigs:
                c.outcome, c.trigger = Outcome.WATCH, trigs[0]
                c.rejections.append(f"setup forming, not confirmed: {trigs[0].note}")
            else:
                c.rejections.append("no valid trigger (sweep+reclaim / breakout-retest / level bounce) at a structural level")
            return c
        trig = valid[0]
        c.trigger = trig
        c.passed.append(f"trigger: {trig.kind} - {trig.note}")
        watch: list[str] = []
        block: list[str] = []
        # 4) confirmations
        if not is_finite(trig.volume_ratio) or trig.volume_ratio < cfg.min_volume_ratio:
            watch.append(f"volume confirmation missing ({trig.volume_ratio:.2f}x < {cfg.min_volume_ratio}x)" if is_finite(trig.volume_ratio) else "volume ratio unavailable")
        else:
            c.passed.append(f"volume confirms ({trig.volume_ratio:.2f}x)")
        s_ok, s_why = self._structure_ok(d, mtf, trig.kind)
        (c.passed if s_ok else watch).append(s_why)
        heat_score, heat_ok, heat_why = heatmap_support(d, inp.heatmap, trig.reclaim)
        if heat_score is None:
            if cfg.liquidation_mode == "required":
                block.append(f"required liquidation context missing: {heat_why}")
            elif "liquidation heatmap unavailable" not in ev.reduced_features:
                ev.reduced_features.append("liquidation heatmap unavailable")
        elif heat_ok:
            c.passed.append(f"liquidation context supports: {heat_why}")
        else:
            watch.append(f"liquidation context does not support {d.value}: {heat_why}")
        if oi_contradiction(d.sign, inp.price_change_pct, inp.oi_change_pct, cfg.oi_contradiction_pct) and not trig.reclaim:
            watch.append(f"OI contradiction: positions added (+{inp.oi_change_pct:.2f}%) while price moved {inp.price_change_pct:+.2f}% against the trade")
        ob = inp.orderbook
        if ob is None:
            (block if cfg.require_orderbook else []).append("order-book data unavailable")
            if not cfg.require_orderbook and "order book unavailable" not in ev.reduced_features:
                ev.reduced_features.append("order book unavailable")
        else:
            if ob.spread_bps > cfg.max_spread_bps:
                block.append(f"spread {ob.spread_bps:.1f} bps > {cfg.max_spread_bps}")
            if min(ob.bid_notional, ob.ask_notional) < cfg.min_depth_notional_usd:
                block.append(f"order-book depth within {cfg.orderbook_depth_pct}% below {cfg.min_depth_notional_usd:,.0f} USDT per side")
        # 5) entry / SL / TP plan (net of costs)
        near = ev.cluster_above if d is Direction.LONG else ev.cluster_below
        plan, plan_why = build_plan(d, trig, ev.atr15, targets, cfg, self.risk, inp.tick_size, inp.funding_rate,
                                    near["distance_pct"] if near else None)
        if plan is None:
            c.rejections += block + watch + plan_why
            return c
        c.plan = plan
        c.passed.append(f"plan valid: net R/R {plan.rr_net:.2f} >= {cfg.min_rr}")
        # 6) score (after hard filters)
        present = [inp.ratio is not None, inp.oi_change_pct is not None, inp.funding_rate is not None, ob is not None,
                   inp.heatmap is not None and inp.heatmap.available]
        dq = sum(present) / len(present)
        ratio_v = inp.ratio.value if inp.ratio else None
        c.components = compute_components(d, mtf, quality, trig.volume_ratio, heat_score,
                                          (inp.price_change_pct, inp.oi_change_pct), ratio_v, inp.funding_rate, ob,
                                          plan.rr_net, dq, cfg)
        c.score = aggregate(c.components)
        if block:
            c.rejections += block + watch
            c.outcome = Outcome.NO_TRADE
        elif watch:
            c.rejections += watch
            c.outcome = Outcome.WATCH
        elif c.score < cfg.min_score:
            c.rejections.append(f"score {c.score:.1f} below minimum {cfg.min_score}")
            c.outcome = Outcome.WATCH
        else:
            c.outcome = Outcome.TRADE_SETUP
        return c
