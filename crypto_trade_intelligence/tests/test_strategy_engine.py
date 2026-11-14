"""Acceptance criteria 6-9 and 15 (+ data validation)."""
import logging
from datetime import timedelta
from decimal import Decimal

import numpy as np

from src.market_structure.trend import analyze_mtf
from src.models import Direction, Outcome, TrendLabel
from src.strategies.planner import Target, build_plan
from src.strategies.setups import Trigger
from tests.helpers import *


def _long_eval(cfg=None, df=None, **kw):
    cfg = cfg or default_cfg()
    df = df if df is not None else build_long_scenario(cfg)[0]
    return engine_for(cfg).evaluate(make_input(df, cfg, **kw)), cfg, df


def test_valid_long_setup_is_a_trade_setup():
    ev, cfg, _ = _long_eval()
    assert ev.outcome is Outcome.TRADE_SETUP and ev.direction is Direction.LONG
    assert ev.trigger.kind == "liquidity_sweep" and ev.score >= cfg.strategy.min_score
    assert "liquidation heatmap unavailable" in ev.reduced_features        # reduced-feature signals are labelled


def test_6_liquidation_cluster_alone_cannot_trigger_a_trade():
    cfg = default_cfg()
    cfg.strategy.liquidation_mode = "required"
    full, _ = build_long_scenario(cfg)
    # (a) same market BEFORE the sweep/reclaim candles: no trigger exists, however large the cluster is
    before = full.iloc[:-7]
    for above in (True, False):
        ev = engine_for(cfg).evaluate(make_input(before, cfg, heatmap=strong_heatmap(cfg, before, above)))
        assert ev.outcome is not Outcome.TRADE_SETUP, ev.rejections
        assert ev.plan is None or ev.outcome is not Outcome.TRADE_SETUP
    # (b) trendless noise with huge clusters on both sides
    rng = np.random.default_rng(3)
    n = 14000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)) * 0.2)
    from tests.helpers import _frame
    flat = _frame(np.r_[c[0], c[:-1]], c * 1.0005, c * 0.9995, c, rng.uniform(80, 120, n))
    for above in (True, False):
        ev = engine_for(cfg).evaluate(make_input(flat, cfg, heatmap=strong_heatmap(cfg, flat, above)))
        assert ev.outcome is not Outcome.TRADE_SETUP, ev.rejections


def test_missing_liquidation_data_blocks_trade_when_required():
    cfg = default_cfg()
    cfg.strategy.liquidation_mode = "required"
    ev, _, _ = _long_eval(cfg)
    assert ev.outcome is Outcome.NO_TRADE
    assert any("liquidation" in r for r in ev.rejections), ev.rejections


def test_full_feature_long_with_heatmap_and_orderbook_can_trade():
    cfg = default_cfg()
    cfg.strategy.liquidation_mode, cfg.strategy.require_orderbook = "required", True
    df, _ = build_long_scenario(cfg)
    from src.models import OrderBookMetrics
    ob = OrderBookMetrics(float(df.close.iloc[-1]), 1.0, 2e6, 1.5e6, 0.14, None)
    ev = engine_for(cfg).evaluate(make_input(df, cfg, heatmap=strong_heatmap(cfg, df, above=True), orderbook=ob,
                                             oi_change=1.2, price_change=0.4, funding=0.0))
    assert ev.outcome is Outcome.TRADE_SETUP, ev.rejections
    assert not ev.reduced_features and ev.cluster_above is not None
    assert ev.components["liquidation"]["score"] is not None
    thin = OrderBookMetrics(float(df.close.iloc[-1]), 1.0, 1e4, 1e4, 0.0, None)         # thin depth -> blocked
    ev2 = engine_for(cfg).evaluate(make_input(df, cfg, heatmap=strong_heatmap(cfg, df, above=True), orderbook=thin))
    assert ev2.outcome is Outcome.NO_TRADE and any("depth" in r for r in ev2.rejections)


def test_7_conflicting_timeframes_produce_wait():
    cfg = default_cfg()
    df, _ = build_long_scenario(cfg)
    inp = make_input(df, cfg)
    inp.candles["4h"] = mirror(inp.candles["4h"])                 # 1h bullish, 4h bearish -> conflict
    mtf = analyze_mtf(inp.candles, cfg.strategy)
    assert mtf.bias == "conflict", (mtf.bias, mtf.labels())
    ev = engine_for(cfg).evaluate(inp)
    assert ev.outcome in (Outcome.WATCH, Outcome.NO_TRADE) and ev.outcome is not Outcome.TRADE_SETUP
    assert ev.outcome is Outcome.WATCH and ev.direction is None
    assert any("conflict" in r for r in ev.rejections)


def test_8_stop_and_targets_come_from_valid_market_data():
    cfg = default_cfg()
    df, info = build_long_scenario(cfg)
    inp = make_input(df, cfg)
    inp.tick_size = Decimal("0.01")
    ev = engine_for(cfg).evaluate(inp)
    p, t = ev.plan, ev.trigger
    assert ev.outcome is Outcome.TRADE_SETUP
    assert all(np.isfinite(x) for x in [p.entry, p.stop_loss, *p.tps])
    assert p.stop_loss < t.invalidation < p.entry                    # beyond structural invalidation
    assert (t.invalidation - p.stop_loss) >= cfg.strategy.stop_atr_buffer * ev.atr15 - 0.011
    assert p.tps == sorted(p.tps) and p.tps[0] > p.entry and 1 <= len(p.tps) <= 3
    assert abs(sum(p.tp_split) - 1) < 1e-9 and len(p.tp_split) == len(p.tps)
    assert p.rr_net >= cfg.strategy.min_rr
    for x in (p.entry, p.stop_loss, *p.tps):                         # exchange tick precision respected
        assert abs(Decimal(str(x)) % Decimal("0.01")) in (Decimal(0), Decimal("0.01")) or abs(round(x, 2) - x) < 1e-9
    assert min(p.stop_atr, 99) >= cfg.strategy.min_stop_atr - 1e-9
    short_trig = Trigger("level_bounce", "valid", 100, 100, 100.5, 100, 2.0, "x")       # wrong side -> rejected
    plan, why = build_plan(Direction.LONG, short_trig, 1.0, [Target(105, "level", 1, "t")], cfg.strategy, cfg.risk, None, 0, None)
    assert plan is None and why


def test_9_invalid_reward_risk_is_rejected():
    cfg = default_cfg()
    cfg.strategy.min_rr = 50.0
    ev, _, _ = _long_eval(cfg)
    assert ev.outcome is Outcome.NO_TRADE and ev.plan is None
    assert any("reward/risk" in r for r in ev.rejections), ev.rejections
    # costs matter: targets that look fine gross can fail net of fees/slippage
    cfg2 = default_cfg()
    trig = Trigger("level_bounce", "valid", 100, 100, 99.8, 100, 2.0, "x")
    targets = [Target(100.55, "level", 1, "near")]
    plan, why = build_plan(Direction.LONG, trig, 0.3, targets, cfg2.strategy, cfg2.risk, None, 0, None)
    assert plan is None and why
    cfg3 = default_cfg()
    cfg3.risk.taker_fee = 0.005                                      # high fees flip a marginal setup to invalid
    ok_plan, _ = build_plan(Direction.LONG, trig, 0.3, [Target(101.5, "level", 1, "a"), Target(102.5, "level", 1, "b")],
                            cfg2.strategy, cfg2.risk, None, 0, None)
    bad_plan, why2 = build_plan(Direction.LONG, trig, 0.3, [Target(101.5, "level", 1, "a"), Target(102.5, "level", 1, "b")],
                                cfg3.strategy, cfg3.risk, None, 0, None)
    assert ok_plan is not None and bad_plan is None and "after fees" in why2[0]


def test_opposing_level_blocking_the_path_is_rejected():
    cfg = default_cfg()
    trig = Trigger("level_bounce", "valid", 100, 100, 99.5, 100, 2.0, "x")
    plan, why = build_plan(Direction.LONG, trig, 0.4, [Target(100.3, "level", 4, "strong resistance")], cfg.strategy, cfg.risk, None, 0, None)
    assert plan is None and "blocks the path" in why[0]


def test_data_problems_never_produce_a_signal():
    cfg = default_cfg()
    df, _ = build_long_scenario(cfg)
    base = make_input(df, cfg)
    eng = engine_for(cfg)
    inp = make_input(df, cfg); inp.price = None
    assert eng.evaluate(inp).outcome is Outcome.NO_TRADE
    inp = make_input(df, cfg); inp.price = float("nan")
    assert eng.evaluate(inp).outcome is Outcome.NO_TRADE
    stale = make_input(df, cfg, now=base.now + timedelta(hours=3))
    ev = eng.evaluate(stale)
    assert ev.outcome is Outcome.NO_TRADE and any("stale" in r for r in ev.rejections)
    bad = df.copy(); bad.iloc[-3, bad.columns.get_loc("close")] = np.nan
    assert eng.evaluate(make_input(bad, cfg)).outcome is Outcome.NO_TRADE
    inp = make_input(df, cfg); inp.candles.pop("4h")
    assert eng.evaluate(inp).outcome is Outcome.NO_TRADE
    inp = make_input(df, cfg, price=float(df.close.iloc[-1]) * 1.3)
    assert eng.evaluate(inp).outcome is Outcome.NO_TRADE


def test_15_every_decision_is_logged_with_reasons():
    records = []

    class H(logging.Handler):
        def emit(self, record): records.append(record)

    lg = logging.getLogger("strategy")
    h = H(); lg.addHandler(h); lg.setLevel(logging.INFO)
    try:
        cfg = default_cfg()
        df, _ = build_long_scenario(cfg)
        engine_for(cfg).evaluate(make_input(df, cfg))                                  # accepted
        cfg.strategy.min_rr = 50.0
        engine_for(cfg).evaluate(make_input(df, cfg))                                  # rejected
    finally:
        lg.removeHandler(h)
    ds = [r for r in records if r.getMessage() == "setup_decision"]
    assert len(ds) == 2
    acc, rej = ds[0].fields, ds[1].fields
    assert acc["outcome"] == "TRADE SETUP" and acc["passed"] and acc["candidates"]["LONG"]["passed"]
    assert rej["outcome"] == "NO TRADE" and any("reward/risk" in x for x in rej["rejections"])
    assert "SHORT" in rej["candidates"] and rej["candidates"]["SHORT"]["rejections"]  # the other direction is explained too


def test_trend_labels_and_structure_sanity():
    cfg = default_cfg()
    df, _ = build_long_scenario(cfg)
    mtf = analyze_mtf(make_input(df, cfg).candles, cfg.strategy)
    assert mtf.frames["4h"].label in (TrendLabel.STRONG_UP, TrendLabel.WEAK_UP)
    assert mtf.bias == "bullish" and 0 < mtf.bias_quality <= 1
    short_df, _ = build_short_scenario(cfg)
    assert analyze_mtf(make_input(short_df, cfg).candles, cfg.strategy).bias == "bearish"
