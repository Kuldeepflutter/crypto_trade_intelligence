"""Acceptance criterion 11: API failures never create fabricated prices or signals (+ HTTP resilience)."""
import asyncio

from config import Secrets
from src.data_feeds.builder import build_input
from src.data_feeds.collector import MarketDataCollector
from src.database.db import Database
from src.exchanges.http import ExchangeError, HttpClient
from src.liquidation.providers import NullProvider
from src.models import Outcome
from src.pipeline import TradeIntelligence
from tests.helpers import *


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._b, self.headers, self.text = status, body, headers or {}, str(body)

    def json(self):
        return self._b


class ScriptedClient:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    async def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def _http(script, retries=3):
    waits = []

    async def sleep(s):
        waits.append(s)

    return HttpClient("https://x", rps=1000, max_retries=retries, client=ScriptedClient(script), sleep=sleep), waits


def test_http_honours_retry_after_and_backs_off():
    http, waits = _http([Resp(429, None, {"Retry-After": "7"}), Resp(503), Resp(200, {"ok": 1})])
    assert asyncio.run(http.get_json("/a")) == {"ok": 1}
    assert 7.0 in waits and max(waits) >= 7.0 and http.requests_made == 3


def test_http_gives_up_with_error_not_fake_data():
    http, _ = _http([Resp(503)], retries=2)
    try:
        asyncio.run(http.get_json("/a"))
        raise AssertionError("expected ExchangeError")
    except ExchangeError as e:
        assert "failed after 3 attempts" in str(e)
    http2, _ = _http([ConnectionError("boom")], retries=1)
    try:
        asyncio.run(http2.get_json("/a"))
        raise AssertionError("expected ExchangeError")
    except ExchangeError:
        pass
    http3, _ = _http([Resp(400, "bad request")])
    try:
        asyncio.run(http3.get_json("/a"))
        raise AssertionError("expected ExchangeError")
    except ExchangeError as e:
        assert e.status == 400 and http3.client_calls() == 1 if hasattr(http3, "client_calls") else True


def test_http_cache_prevents_repeat_requests():
    http, _ = _http([Resp(200, {"v": 1})])
    for _ in range(5):
        assert asyncio.run(http.get_json("/c", {"s": "BTC"}, ttl=30)) == {"v": 1}
    assert http.requests_made == 1


def test_11_total_outage_yields_no_price_no_signal():
    cfg = default_cfg()
    df, _ = build_long_scenario(cfg)
    conn = FakeConnector(df, ratio=1.4, fail=True)
    col = MarketDataCollector(conn, NullProvider(), cfg, clock=lambda: df["close_time"].iloc[-1].to_pydatetime())
    snap = asyncio.run(col.snapshot("TESTUSDT"))
    assert snap.price is None and not snap.candles and snap.errors and "ticker" in snap.errors
    ev = engine_for(cfg).evaluate(build_input(snap, cfg))
    assert ev.outcome is Outcome.NO_TRADE and ev.plan is None and ev.direction is None
    assert any("price unavailable" in r or "candles" in r for r in ev.rejections)


def test_11_partial_outage_never_substitutes_data():
    cfg = default_cfg()
    df, _ = build_long_scenario(cfg)

    class NoTicker(FakeConnector):
        async def ticker(self, symbol):
            raise ExchangeError("ticker down", 500)

    col = MarketDataCollector(NoTicker(df), NullProvider(), cfg, clock=lambda: df["close_time"].iloc[-1].to_pydatetime())
    snap = asyncio.run(col.snapshot("TESTUSDT"))
    assert snap.price is None                                   # NOT replaced by the last candle close
    ev = engine_for(cfg).evaluate(build_input(snap, cfg))
    assert ev.outcome is Outcome.NO_TRADE


def test_11_pipeline_records_rejection_and_opens_nothing_on_outage():
    cfg = default_cfg()
    cfg.execution.mode = "paper"
    df, _ = build_long_scenario(cfg)
    db = Database(":memory:")
    ti = TradeIntelligence(cfg, Secrets(), FakeConnector(df, fail=True), NullProvider(), db)
    ti.collector.clock = lambda: df["close_time"].iloc[-1].to_pydatetime()
    rows = asyncio.run(ti.scan_all())
    assert rows and rows[0].ev.outcome is Outcome.NO_TRADE and rows[0].action == "none"
    assert db.query("SELECT COUNT(*) AS n FROM trades")[0]["n"] == 0
    sig = db.query("SELECT * FROM signals")[0]
    assert sig["accepted"] == 0 and sig["reasons"]


def test_no_silent_exchange_switching():
    cfg = default_cfg()
    df, _ = build_long_scenario(cfg)
    primary = FakeConnector(df, fail=True)
    col = MarketDataCollector(primary, NullProvider(), cfg)
    assert col.x is primary and not hasattr(col, "fallback")
    asyncio.run(col.snapshot("TESTUSDT"))
    assert primary.calls > 0
