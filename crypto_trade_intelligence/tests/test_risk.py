"""Acceptance criterion 10: position size respects the risk budget and exchange limits (+ kill switch)."""
from datetime import datetime, timezone
from decimal import Decimal

from config import RiskConfig
from src.models import Direction, SymbolRules
from src.risk.manager import RiskManager
from src.risk.sizing import liquidation_price, size_position, stop_inside_liquidation

RULES = SymbolRules("BTCUSDT", Decimal("0.1"), Decimal("0.001"), Decimal("0.001"), Decimal("100"))
NOW = datetime(2025, 6, 1, 12, tzinfo=timezone.utc)


def _size(entry=60000.0, stop=59400.0, equity=10000.0, risk=0.5, lev=5.0, alloc=20.0, rules=RULES, exposure=None):
    return size_position(entry, stop, equity, risk, lev, rules, alloc, 0.0005, 0.05, exposure)


def test_10_size_never_exceeds_risk_budget_and_respects_step():
    for entry, stop in [(60000, 59400), (60000, 59990), (3000, 2940), (0.5, 0.49), (150, 148.5), (60000, 61000)]:
        for risk in (0.25, 0.5, 1.0, 2.0):
            rules = RULES if entry > 1000 else SymbolRules("X", Decimal("0.0001"), Decimal("1"), Decimal("1"), Decimal("5"))
            s = size_position(entry, stop, 10000, risk, 5, rules, 20, 0.0005, 0.05)
            if not s.ok:
                continue
            assert s.risk_actual <= s.risk_budget + 1e-9, (entry, stop, risk, s)
            assert s.quantity % rules.step_size == 0 and s.quantity >= rules.min_qty and s.notional >= rules.min_notional
            assert float(s.margin) <= 10000 * 0.20 + 1e-6
    s = _size()
    assert s.ok and s.risk_budget == 50.0 and s.risk_actual <= 50.0 and s.risk_actual > 45.0


def test_leverage_does_not_raise_the_risk_budget():
    a, b = _size(lev=5), _size(lev=20)
    assert a.ok and b.ok and a.quantity == b.quantity and a.risk_actual == b.risk_actual     # risk-bound -> same size
    low = _size(lev=2)                                                                        # margin-bound: smaller, never larger
    assert low.ok and low.quantity <= a.quantity and low.risk_actual <= low.risk_budget
    capped = _size(stop=59990.0, lev=2, alloc=10)                                            # tight stop: margin cap binds
    assert capped.ok and float(capped.margin) <= 10000 * 0.10 + 1e-6 and capped.risk_actual < capped.risk_budget


def test_exchange_limits_reject_instead_of_exceeding_risk():
    assert _size(rules=None).rejected and "rules" in _size(rules=None).rejected
    big_min = SymbolRules("BTCUSDT", Decimal("0.1"), Decimal("0.001"), Decimal("0.5"), Decimal("100"))   # min 0.5 BTC = 30k notional
    s = _size(rules=big_min)
    assert not s.ok and "minimum" in s.rejected
    assert _size(entry=float("nan")).rejected and _size(stop=60000.0).rejected                # invalid / zero stop distance
    assert _size(exposure=0.0).rejected                                                       # no exposure headroom


def test_liquidation_distance_check():
    liq = liquidation_price(Direction.LONG, 100, 10, 0.005)
    assert abs(liq - 90.5) < 1e-9
    assert stop_inside_liquidation(Direction.LONG, 100, 97, 10, 0.005, 0.8)
    assert not stop_inside_liquidation(Direction.LONG, 100, 92, 10, 0.005, 0.8)
    assert liquidation_price(Direction.SHORT, 100, 10, 0.005) > 100


def test_kill_switch_blocks_new_entries_after_daily_loss_limit():
    cfg = RiskConfig(account_equity=10000, max_daily_loss_pct=2.0, max_positions=2)
    pnl = {"v": -150.0}
    rm = RiskManager(cfg, lambda now: pnl["v"], lambda: pnl["v"])
    assert rm.approve_entry([], 1000, NOW).allowed
    pnl["v"] = -200.0                                                    # exactly 2% of 10k
    killed, why = rm.kill_switch_active(NOW)
    assert killed and "KILL SWITCH" in why
    d = rm.approve_entry([], 1000, NOW)
    assert not d.allowed and any("KILL SWITCH" in r for r in d.reasons)


def test_position_count_and_exposure_limits():
    cfg = RiskConfig(account_equity=10000, max_positions=2, max_total_exposure_pct=100)
    rm = RiskManager(cfg)
    assert not rm.approve_entry([1000, 1000], 100, NOW).allowed
    assert not rm.approve_entry([9500], 1000, NOW).allowed
    assert rm.approve_entry([1000], 1000, NOW).allowed
