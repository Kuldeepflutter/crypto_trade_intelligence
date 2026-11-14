"""Acceptance criteria 1-5: the mandatory SHORT filter (long/short ratio >= 1.0)."""
from datetime import datetime, timedelta, timezone

from config import SHORT_MIN_RATIO
from src.models import Direction, Outcome, RatioKind, RatioReading
from src.positioning.ratio import ratio_gate, short_entry_allowed
from tests.helpers import *


def _short_eval(ratio_value, age=60, kind=RatioKind.GLOBAL_ACCOUNT, df=None, cfg=None):
    cfg = cfg or default_cfg()
    df = df if df is not None else build_short_scenario(cfg)[0]
    now = df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    ratio = None if ratio_value is None else ratio_reading(ratio_value, now, kind, age)
    ev = engine_for(cfg).evaluate(make_input(df, cfg, ratio=ratio))
    return ev, ev.candidates["SHORT"]


def test_threshold_is_fixed_at_one():
    assert SHORT_MIN_RATIO == 1.0


def test_1_short_with_ratio_085_is_rejected():
    assert short_entry_allowed(0.85) is False
    ev, c = _short_eval(0.85)
    assert c.outcome is Outcome.NO_TRADE and ev.outcome is not Outcome.TRADE_SETUP
    assert any("0.850" in r and "< 1.0" in r for r in c.rejections), c.rejections
    assert c.plan is None and c.score is None          # blocked BEFORE any signal/plan is generated


def test_2_short_with_ratio_099_is_rejected():
    assert short_entry_allowed(0.99) is False
    ev, c = _short_eval(0.99)
    assert c.outcome is Outcome.NO_TRADE and c.plan is None


def test_3_ratio_exactly_1_passes_only_the_gate():
    assert short_entry_allowed(1.0) is True
    cfg = default_cfg()
    df = flatten_volume(build_short_scenario(cfg)[0])            # remove the volume confirmation
    ev, c = _short_eval(1.0, df=df, cfg=cfg)
    assert any("ratio gate passed" in p for p in c.passed)       # gate passed...
    assert c.outcome is not Outcome.TRADE_SETUP                  # ...but that alone authorizes nothing
    assert any("volume" in r for r in c.rejections), c.rejections


def test_4_missing_stale_or_invalid_ratio_is_rejected():
    cfg = default_cfg()
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    k = RatioKind.GLOBAL_ACCOUNT
    cases = {
        "missing": None,
        "none value": RatioReading(None, k, now, "t"),
        "nan": RatioReading(float("nan"), k, now, "t"),
        "inf": RatioReading(float("inf"), k, now, "t"),
        "zero": RatioReading(0.0, k, now, "t"),
        "negative": RatioReading(-1.0, k, now, "t"),
        "no timestamp": RatioReading(1.5, k, None, "t"),
        "stale": RatioReading(1.5, k, now - timedelta(seconds=cfg.ratio.max_age_seconds + 60), "t"),
        "future timestamp": RatioReading(1.5, k, now + timedelta(hours=1), "t"),
        "wrong kind": RatioReading(1.5, RatioKind.TOP_TRADER_POSITION, now, "t"),
    }
    for name, reading in cases.items():
        g = ratio_gate(reading, k, now, cfg.ratio.max_age_seconds)
        assert g.passed is False, name
    assert ratio_gate(RatioReading(1.5, k, now, "t"), k, now, cfg.ratio.max_age_seconds).passed is True
    for bad in (None, float("nan"), float("inf"), True, "1.5"):
        assert short_entry_allowed(bad) is False
    # same through the engine
    for value, age, kind in ((None, 60, k), (1.5, 3600, k), (1.5, 60, RatioKind.TOP_TRADER_ACCOUNT)):
        ev, c = _short_eval(value, age, kind)
        assert c.outcome is Outcome.NO_TRADE and c.plan is None, (value, age, kind)


def test_ratio_above_one_with_qualifying_setup_can_trade():
    ev, c = _short_eval(1.4)
    assert c.outcome is Outcome.TRADE_SETUP and ev.direction is Direction.SHORT
    assert c.plan.stop_loss > c.plan.entry > max(c.plan.tps)


def test_5_long_is_not_auto_approved_by_a_short_rejection():
    cfg = default_cfg()
    bear = build_short_scenario(cfg)[0]                                   # bearish market, SHORT blocked by ratio
    ev, short_c = _short_eval(0.8, df=bear, cfg=cfg)
    long_c = ev.candidates["LONG"]
    assert short_c.outcome is Outcome.NO_TRADE
    assert long_c.outcome is not Outcome.TRADE_SETUP and ev.outcome is not Outcome.TRADE_SETUP
    assert any("bias" in r for r in long_c.rejections), long_c.rejections
    # and the reverse: a valid LONG market is approved on its own evidence (ratio only feeds the LONG *score*)
    bull = build_long_scenario(cfg)[0]
    now = bull["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    ev2 = engine_for(cfg).evaluate(make_input(bull, cfg, ratio=ratio_reading(0.8, now)))
    assert ev2.candidates["LONG"].outcome is Outcome.TRADE_SETUP
    assert ev2.candidates["SHORT"].outcome is not Outcome.TRADE_SETUP
    assert any("SHORT blocked" in r for r in ev2.candidates["SHORT"].rejections)
    ev3 = engine_for(cfg).evaluate(make_input(bull, cfg, ratio=ratio_reading(3.0, now)))      # crowded longs: lower LONG score
    assert not any("SHORT blocked" in r for r in ev3.candidates["LONG"].rejections)           # the SHORT gate never leaks into LONG
