"""Acceptance criterion 12 (+ SHORT gate at alert time, alert content)."""
import asyncio
from datetime import timedelta

from src.alerts.telegram import AlertManager, format_alert
from src.database.db import Database
from src.models import Direction, Outcome, RatioKind
from tests.helpers import *


class Spy:
    def __init__(self): self.sent = []

    async def send(self, text):
        self.sent.append(text)
        return True


def _mgr(cfg=None):
    cfg = cfg or default_cfg()
    spy = Spy()
    return AlertManager(cfg, Database(":memory:"), spy), spy, cfg


def _long_ev(cfg):
    df, _ = build_long_scenario(cfg)
    return engine_for(cfg).evaluate(make_input(df, cfg))


def test_12_duplicate_alerts_are_suppressed_and_cooldown_expires():
    mgr, spy, cfg = _mgr()
    ev = _long_ev(cfg)
    assert ev.outcome is Outcome.TRADE_SETUP
    now = ev.timestamp
    assert asyncio.run(mgr.maybe_alert(ev, now))[0] is True
    sent, why = asyncio.run(mgr.maybe_alert(ev, now + timedelta(minutes=5)))
    assert sent is False and "duplicate" in why and len(spy.sent) == 1
    assert asyncio.run(mgr.maybe_alert(ev, now + timedelta(minutes=cfg.alerts.cooldown_minutes + 1)))[0] is True
    assert len(spy.sent) == 2


def test_meaningful_change_bypasses_cooldown():
    mgr, spy, cfg = _mgr()
    ev = _long_ev(cfg)
    asyncio.run(mgr.maybe_alert(ev, ev.timestamp))
    ev.score = (ev.score or 0) + cfg.alerts.min_score_change + 1
    assert asyncio.run(mgr.maybe_alert(ev, ev.timestamp + timedelta(minutes=2)))[0] is True
    assert len(spy.sent) == 2


def test_non_setups_are_never_alerted():
    mgr, spy, cfg = _mgr()
    ev = _long_ev(cfg)
    ev.outcome = Outcome.WATCH
    sent, why = asyncio.run(mgr.maybe_alert(ev, ev.timestamp))
    assert sent is False and not spy.sent and "not a qualified" in why


def test_short_below_one_is_never_alerted_even_if_marked_as_setup():
    mgr, spy, cfg = _mgr()
    df, _ = build_short_scenario(cfg)
    now = df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    ev = engine_for(cfg).evaluate(make_input(df, cfg, ratio=ratio_reading(1.4, now)))
    assert ev.outcome is Outcome.TRADE_SETUP and ev.direction is Direction.SHORT
    ev.ratio = ratio_reading(0.9, now)                                     # ratio dropped / tampered after the fact
    sent, why = asyncio.run(mgr.maybe_alert(ev, now))
    assert sent is False and not spy.sent and "SHORT blocked" in why
    ev.ratio = ratio_reading(1.4, now)
    assert asyncio.run(mgr.maybe_alert(ev, now + timedelta(hours=2)))[0] is False      # ratio now stale
    assert not spy.sent
    assert asyncio.run(mgr.maybe_alert(ev, now))[0] is True and len(spy.sent) == 1     # fresh and >= 1.0 -> allowed


def test_alert_contains_required_fields_and_is_honest():
    cfg = default_cfg()
    ev = _long_ev(cfg)
    text = format_alert(ev)
    for f in ("Symbol:", "Direction:", "Current price:", "Entry zone:", "Stop-loss:", "TP1 / TP2 / TP3:", "Reward-to-risk",
              "Long/short ratio:", "OI change:", "Trend alignment:", "Nearest liquidation cluster:", "Setup score:",
              "Data timestamp:", "Reasons supporting the trade:", "Invalidation conditions:"):
        assert f in text, f
    assert "not a win probability" in text and "Reduced-feature" in text
    assert "guarantee" not in text.lower()
