"""Acceptance criterion 14 (paper trading never places real orders) + live-order gates."""
import asyncio
from datetime import timedelta

from config import AppConfig, Secrets
from src.database.db import Database
from src.execution.live import LiveBroker, LiveTradingDisabled
from src.execution.paper import PaperBroker
from src.execution.simulator import PositionSimulator
from src.liquidation.providers import NullProvider
from src.models import Direction, Outcome
from src.pipeline import TradeIntelligence
from src.risk.manager import RiskManager
from src.risk.sizing import size_position
from tests.helpers import *


def _setup(mode="paper", short=False):
    cfg = default_cfg()
    cfg.execution.mode = mode
    df, _ = (build_short_scenario(cfg) if short else build_long_scenario(cfg))
    return cfg, df


def test_default_mode_is_analysis_only_and_live_is_gated():
    cfg = AppConfig()
    assert cfg.execution.mode == "analysis" and cfg.execution.exchange_testnet is True
    assert Secrets.from_env().allow_live_orders is False
    for kw in (dict(mode="analysis"), dict(mode="live")):
        c = AppConfig(); c.execution.mode = kw["mode"]
        try:
            LiveBroker(c, Secrets(binance_api_key="k", binance_api_secret="s", allow_live_orders=False), RiskManager(c.risk))
            raise AssertionError("live broker must refuse")
        except LiveTradingDisabled:
            pass
    c = AppConfig(); c.execution.mode = "live"
    try:
        LiveBroker(c, Secrets(allow_live_orders=True), RiskManager(c.risk))      # no API keys
        raise AssertionError("must refuse without keys")
    except LiveTradingDisabled:
        pass


def test_14_paper_pipeline_opens_and_closes_without_any_exchange_order():
    cfg, df = _setup("paper")
    db = Database(":memory:")
    conn = FakeConnector(df, ratio=1.4)
    ti = TradeIntelligence(cfg, Secrets(), conn, NullProvider(), db)
    ti.collector.clock = lambda: df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    assert ti.live is None and not hasattr(ti.paper, "ex")                       # no exchange client anywhere in paper mode
    row = asyncio.run(ti.scan_symbol("TESTUSDT"))
    assert row.ev.outcome is Outcome.TRADE_SETUP and row.action.startswith("paper_opened"), (row.action, row.notes)
    assert row.sizing.ok and row.sizing.risk_actual <= row.sizing.risk_budget
    trades = db.query("SELECT * FROM trades")
    assert len(trades) == 1 and trades[0]["status"] == "open" and trades[0]["mode"] == "paper"
    asyncio.run(ti.scan_symbol("TESTUSDT"))                                      # same signal again
    assert len(ti.paper.positions) == 1 and db.query("SELECT COUNT(*) AS n FROM trades")[0]["n"] == 1   # no duplicate position
    # price rallies through every target -> position closes with a profit and fees were charged
    rallied = extend(df, 150, +0.12)
    closed = ti.paper.update("TESTUSDT", tf_candles(rallied)["5m"])
    t = db.query("SELECT * FROM trades")[0]
    assert closed and t["status"] == "closed" and t["pnl"] > 0 and t["fees"] > 0 and t["r_multiple"] > 0
    assert t["mfe"] > 0 and t["exit_reason"].startswith("tp")
    assert db.realized_total("paper") == t["pnl"]


def test_paper_stop_loss_costs_more_than_one_r():
    cfg, df = _setup("paper")
    db = Database(":memory:")
    ti = TradeIntelligence(cfg, Secrets(), FakeConnector(df, ratio=1.4), NullProvider(), db)
    ti.collector.clock = lambda: df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    asyncio.run(ti.scan_symbol("TESTUSDT"))
    crashed = extend(df, 150, -0.12)
    ti.paper.update("TESTUSDT", tf_candles(crashed)["5m"])
    t = db.query("SELECT * FROM trades")[0]
    assert t["status"] == "closed" and t["exit_reason"] == "stop_loss" and t["pnl"] < 0
    assert t["r_multiple"] < -1.0 and t["mae"] > 0                                # fees + slippage push the loss beyond 1R


def test_paper_kill_switch_stops_new_entries_after_loss_limit():
    cfg, df = _setup("paper")
    cfg.risk.max_daily_loss_pct = 0.05                                            # tiny limit: one stop-out trips it
    db = Database(":memory:")
    ti = TradeIntelligence(cfg, Secrets(), FakeConnector(df, ratio=1.4), NullProvider(), db)
    now = df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    ti.collector.clock = lambda: now
    asyncio.run(ti.scan_symbol("TESTUSDT"))
    ti.paper.update("TESTUSDT", tf_candles(extend(df, 150, -0.12))["5m"])
    assert db.query("SELECT status FROM trades")[0]["status"] == "closed"
    ev = engine_for(cfg).evaluate(make_input(df, cfg))
    sz = size_position(ev.plan.entry, ev.plan.stop_loss, 10000, 0.5, 5, FakeConnector(df).rules, 20, 0.0005, 0.05)
    later = now + timedelta(minutes=1)                                            # same UTC day as the realized loss
    assert ti.risk.kill_switch_active(later)[0]
    tid, reasons = ti.paper.submit(ev, sz, later, None, None, ev.plan.entry)
    assert tid is None and any("KILL SWITCH" in r for r in reasons)


def test_paper_broker_rejects_short_below_one_even_if_marked_setup():
    cfg, df = _setup("paper", short=True)
    now = df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    ev = engine_for(cfg).evaluate(make_input(df, cfg, ratio=ratio_reading(1.4, now)))
    assert ev.outcome is Outcome.TRADE_SETUP and ev.direction is Direction.SHORT
    db = Database(":memory:")
    broker = PaperBroker(cfg, db, RiskManager(cfg.risk))
    sz = size_position(ev.plan.entry, ev.plan.stop_loss, 10000, 0.5, 5, FakeConnector(df).rules, 20, 0.0005, 0.05)
    tid, reasons = broker.submit(ev, sz, now, None, fresh_ratio=ratio_reading(0.95, now))   # fresh data says 0.95
    assert tid is None and any("SHORT blocked" in r for r in reasons) and not db.query("SELECT * FROM trades")
    tid, reasons = broker.submit(ev, sz, now, None, fresh_ratio=ratio_reading(1.2, now))
    assert tid is not None


def test_simulator_stop_first_and_gap_rules():
    sim = PositionSimulator(0.0005, 0.0002, 0.05)
    t0 = __import__("datetime").datetime(2025, 1, 1)
    p = sim.open("X", Direction.LONG, t0, 100.0, 1.0, 98.0, [103.0], [1.0])
    assert p.entry_price > 100.0                                                   # adverse entry slippage
    sim.on_candle(p, t0, 100, 104, 97, 101)                                        # touches BOTH stop and TP in one bar
    assert p.exit_reason == "stop_loss" and p.pnl < 0                              # stop assumed first
    q = sim.open("X", Direction.LONG, t0, 100.0, 1.0, 98.0, [103.0], [1.0])
    sim.on_candle(q, t0, 95, 96, 94, 95)                                           # gaps through the stop
    assert q.exits[0][1] < 98.0                                                    # filled at the worse open
    s = sim.open("X", Direction.SHORT, t0, 100.0, 1.0, 102.0, [97.0], [1.0])
    sim.on_candle(s, t0, 100, 100.5, 96, 97)
    assert s.exit_reason == "tp1" and s.pnl > 0


class FakeEx:
    def __init__(self, fail_stop=False):
        self.orders, self.fail_stop, self.ticker_last = [], fail_stop, None

    async def fetch_ticker(self, sym): return {"last": self.ticker_last}
    def amount_to_precision(self, sym, a): return f"{a:.3f}"
    def price_to_precision(self, sym, p): return f"{p:.2f}"
    async def set_margin_mode(self, *a): return None
    async def set_leverage(self, *a): return None
    async def close(self): return None

    async def create_order(self, sym, typ, side, amt, price=None, params=None):
        if self.fail_stop and typ == "STOP_MARKET":
            raise RuntimeError("exchange rejected stop")
        self.orders.append((typ, side, amt, price, params or {}))
        return {"id": len(self.orders)}


def _live(short=False, fail_stop=False):
    cfg, df = _setup("live", short)
    now = df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    ev = engine_for(cfg).evaluate(make_input(df, cfg, ratio=ratio_reading(1.4, now) if short else None))
    assert ev.outcome is Outcome.TRADE_SETUP
    ex = FakeEx(fail_stop)
    ex.ticker_last = ev.plan.entry
    broker = LiveBroker(cfg, Secrets(allow_live_orders=True), RiskManager(cfg.risk), exchange=ex)
    sz = size_position(ev.plan.entry, ev.plan.stop_loss, 10000, 0.5, 5, FakeConnector(df).rules, 20, 0.0005, 0.05)
    return broker, ev, sz, now, ex


def test_live_places_reduce_only_protective_orders_when_all_gates_pass():
    broker, ev, sz, now, ex = _live()
    asyncio.run(broker.place_bracket(ev, sz, now, [], None))
    kinds = [o[0] for o in ex.orders]
    assert kinds[0] == "market" and "STOP_MARKET" in kinds and kinds.count("limit") == len(ev.plan.tps)
    assert all(o[4].get("reduceOnly") for o in ex.orders[1:])


def test_live_short_below_one_never_reaches_the_exchange():
    broker, ev, sz, now, ex = _live(short=True)
    try:
        asyncio.run(broker.place_bracket(ev, sz, now, [], ratio_reading(0.9, now)))     # fresh ratio re-fetched: 0.9
        raise AssertionError("must be blocked")
    except LiveTradingDisabled as e:
        assert "SHORT blocked" in str(e)
    assert ex.orders == []


def test_live_flattens_if_protective_stop_fails():
    broker, ev, sz, now, ex = _live(fail_stop=True)
    try:
        asyncio.run(broker.place_bracket(ev, sz, now, [], None))
        raise AssertionError("must raise")
    except RuntimeError:
        pass
    assert ex.orders[0][0] == "market" and ex.orders[-1][0] == "market" and ex.orders[-1][4].get("reduceOnly")
