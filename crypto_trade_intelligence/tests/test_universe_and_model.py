"""All-coins scanning (public data only) and the keyless liquidation model."""
import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import AppConfig, PublicModelConfig, Secrets
from src.data_feeds.ws_feed import LiveFeed
from src.database.db import Database
from src.exchanges.binance import kline_weight, parse_tickers, parse_universe
from src.exchanges.http import HttpClient, WeightLimiter
from src.liquidation.model import PublicModelProvider, estimate_clusters
from src.liquidation.heatmap import analyze_clusters
from src.models import Outcome, RatioKind
from src.pipeline import TradeIntelligence
from src.universe import prioritize, select_candidates, triage_symbol
from tests.helpers import *

NOW = datetime(2025, 3, 1, tzinfo=timezone.utc)


# ------------------------------------------------------------------ rate limiting at scale
def test_weight_limiter_waits_instead_of_exceeding_budget():
    clock = {"t": 0.0}
    waits = []

    async def sleep(s):
        waits.append(s)
        clock["t"] += s

    lim = WeightLimiter(10, 60, clock=lambda: clock["t"], sleep=sleep)

    async def go():
        for _ in range(4):
            await lim.acquire(4)                       # 4+4 fit; the 3rd must wait for the window to slide

    asyncio.run(go())
    assert waits and clock["t"] >= 60.0 and lim._used <= 10
    assert kline_weight(50) == 1 and kline_weight(250) == 2 and kline_weight(500) == 5 and kline_weight(1500) == 10


def test_http_client_charges_request_weight():
    class R:
        status_code, headers, text = 200, {}, ""
        def json(self): return {}

    class C:
        async def get(self, *a, **k): return R()

    http = HttpClient("https://x", rps=1000, client=C(), weight_per_minute=1800)
    asyncio.run(http.get_json("/a", weight=40))
    assert http.budgets["fapi"]._used == 40
    asyncio.run(http.get_json("/b", weight=1, bucket="data"))
    assert http.budgets["data"]._used == 1


# ------------------------------------------------------------------ universe discovery / filtering
def test_universe_contains_only_trading_usdt_perpetuals():
    info = {"symbols": [
        {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING", "onboardDate": 1569398400000},
        {"symbol": "BTCUSDT_250627", "contractType": "CURRENT_QUARTER", "quoteAsset": "USDT", "status": "TRADING"},
        {"symbol": "ETHUSDC", "contractType": "PERPETUAL", "quoteAsset": "USDC", "status": "TRADING"},
        {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "SETTLING"},
        {"symbol": "NEWUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "PENDING_TRADING"}]}
    assert [u["symbol"] for u in parse_universe(info)] == ["BTCUSDT"]
    t = parse_tickers([{"symbol": "BTCUSDT", "lastPrice": "60000", "quoteVolume": "9e9", "priceChangePercent": "1.5"}])
    assert t.at["BTCUSDT", "quote_volume"] == 9e9


def test_candidate_filters_volume_age_exclusions_and_cap():
    cfg = AppConfig()
    cfg.universe.max_symbols = 3
    cfg.strategy.min_volume_24h_usd = 20e6
    uni = [{"symbol": s, "onboard": NOW - timedelta(days=d)} for s, d in
           [("AAAUSDT", 400), ("BBBUSDT", 400), ("CCCUSDT", 400), ("DDDUSDT", 400), ("NEWUSDT", 5), ("LOWUSDT", 400), ("USDCUSDT", 400), ("NOTICKUSDT", 400)]]
    vols = {"AAAUSDT": 5e9, "BBBUSDT": 4e9, "CCCUSDT": 3e9, "DDDUSDT": 2e9, "NEWUSDT": 9e9, "LOWUSDT": 1e6, "USDCUSDT": 8e9}
    tick = pd.DataFrame({"last": 1.0, "quote_volume": list(vols.values()), "change_pct": 0.0}, index=list(vols))
    syms, skipped = select_candidates(uni, tick, cfg, NOW)
    assert syms == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]                      # highest volume first, capped
    assert skipped["excluded"] == 1 and skipped["24h volume below minimum"] == 1 and skipped["no ticker"] == 1
    assert any("listed <" in k for k in skipped) and any("max_symbols" in k for k in skipped)
    cfg.universe.max_symbols, cfg.strategy.min_volume_24h_usd = 100, 0.0                      # "every coin" setting
    all_syms, _ = select_candidates(uni, tick, cfg, NOW)
    assert "LOWUSDT" in all_syms and "NEWUSDT" not in all_syms


def test_rotation_gives_every_coin_a_deep_scan_and_keeps_priorities():
    passed = [f"C{i}" for i in range(10)]
    seen, cursor = set(), 0
    for _ in range(4):
        deep, cursor = prioritize(passed, ["HELD"], 4, cursor)
        assert deep[0] == "HELD" and len(deep) == 4
        seen |= set(deep) - {"HELD"}
    assert seen == set(passed)


# ------------------------------------------------------------------ end-to-end all-coins scan
def _all_cfg():
    cfg = default_cfg()
    cfg.universe.mode = "all"
    cfg.universe.max_deep_scans_per_cycle = 10
    return cfg


def test_all_coins_scan_triages_then_deep_scans_only_what_matters():
    cfg = _all_cfg()
    bull, _ = build_long_scenario(cfg)
    flat = noise_frame()
    frames = {"UPAUSDT": bull, "UPBUSDT": bull, "FLATUSDT": flat, "FLT2USDT": noise_frame(seed=4),
              "THINUSDT": bull, "NEWUSDT": bull, "USDCUSDT": bull}
    conn = UniverseConnector(frames, volumes={"THINUSDT": 1e6}, onboard_days={"NEWUSDT": 3})
    ti = TradeIntelligence(cfg, Secrets(), conn, __import__("src.liquidation.providers", fromlist=["x"]).NullProvider(), Database(":memory:"))
    ti.collector.clock = lambda: bull["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    rows = asyncio.run(ti.scan_all())
    got = {r.symbol: r for r in rows}
    assert set(got) == {"UPAUSDT", "UPBUSDT"}, (set(got), {s: t.reason for s, t in ti.triage.items()})
    assert all(r.ev.outcome is Outcome.TRADE_SETUP for r in rows)
    st = ti.universe_stats
    assert st.total == 7 and st.candidates == 4 and st.triaged_pass == 2 and st.deep_scanned == 2
    assert st.skipped["excluded"] == 1 and st.skipped["24h volume below minimum"] == 1
    assert ("FLATUSDT", "1h") in conn.kline_calls and ("FLATUSDT", "5m") not in conn.kline_calls   # skipped coins cost only 1h/4h calls
    assert ("UPAUSDT", "5m") in conn.kline_calls and ("THINUSDT", "1h") not in conn.kline_calls
    before = conn.kline_calls[("FLATUSDT", "1h")]
    asyncio.run(ti.scan_all())
    assert conn.kline_calls[("FLATUSDT", "1h")] == before                  # triage cached, not re-fetched every cycle


def test_all_coins_scan_survives_total_outage_without_signals():
    cfg = _all_cfg()
    bull, _ = build_long_scenario(cfg)
    conn = UniverseConnector({"UPAUSDT": bull})
    conn.fail = True

    async def boom():
        raise RuntimeError("exchange unreachable")
    conn.all_tickers = boom
    ti = TradeIntelligence(cfg, Secrets(), conn, __import__("src.liquidation.providers", fromlist=["x"]).NullProvider(), Database(":memory:"))
    assert asyncio.run(ti.scan_all()) == [] and "unreachable" in ti.last_error
    assert ti.db.query("SELECT COUNT(*) AS n FROM signals")[0]["n"] == 0


def test_triage_unreadable_coin_is_skipped_not_guessed():
    cfg = _all_cfg()
    conn = UniverseConnector({"UPAUSDT": build_long_scenario(cfg)[0]})
    conn.fail = True
    r = asyncio.run(triage_symbol(conn, "UPAUSDT", cfg, NOW))
    assert r.passed is False and "data error" in r.reason


def test_journal_records_state_changes_not_every_scan():
    cfg = _all_cfg()
    bull, _ = build_long_scenario(cfg)
    flat = noise_frame()
    conn = UniverseConnector({"UPAUSDT": bull})
    ti = TradeIntelligence(cfg, Secrets(), conn, __import__("src.liquidation.providers", fromlist=["x"]).NullProvider(), Database(":memory:"))
    ti.collector.clock = lambda: bull["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    ti.cfg.universe.mode = "fixed"
    ti.cfg.symbols = ["UPAUSDT"]
    cfg.strategy.min_rr = 99                                   # permanently rejected -> same state every scan
    ti.engine.cfg.min_rr = 99
    for _ in range(3):
        asyncio.run(ti.scan_symbol("UPAUSDT"))
    assert ti.db.query("SELECT COUNT(*) AS n FROM signals")[0]["n"] == 1


# ------------------------------------------------------------------ keyless liquidation model
def _model_inputs(n=300, oi_growth=1.0, ratio_val=2.0, crash_at=None, crash_depth=0.03):
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    price = np.full(n, 100.0)
    if crash_at is not None:
        price[crash_at:] = 100.0 * (1 - crash_depth)
    c = pd.DataFrame({"open": price, "high": price * 1.001, "low": price * 0.999, "close": price, "volume": 1.0}, index=idx)
    c["close_time"] = idx + pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1)
    oi = pd.DataFrame({"oi": 1000.0 + oi_growth * np.arange(n)}, index=idx + pd.Timedelta(minutes=15))
    ratio = pd.DataFrame({"ratio": np.full(n, ratio_val)}, index=idx + pd.Timedelta(minutes=15))
    return c, oi, ratio


def test_model_builds_clusters_on_the_right_sides_and_is_labelled_estimated():
    c, oi, ratio = _model_inputs(ratio_val=3.0)                         # 75% longs
    res = estimate_clusters(c, oi, ratio, PublicModelConfig(), 0.005, 15)
    assert res.reason == "ok" and res.clusters
    assert all(cl.estimated and cl.source.startswith("public_model") and "estimate" in cl.source for cl in res.clusters)
    longs = [cl for cl in res.clusters if cl.side == "long"]
    shorts = [cl for cl in res.clusters if cl.side == "short"]
    assert all(cl.price < 100 for cl in longs) and all(cl.price > 100 for cl in shorts)
    assert sum(cl.notional_usd for cl in longs) > 2 * sum(cl.notional_usd for cl in shorts)     # ratio 3 -> mostly longs
    assert res.info["surviving_usd"] <= res.info["added_usd"] + 1e-6                            # exposure never created from nothing


def test_model_removes_exposure_that_has_already_been_liquidated():
    c, oi, ratio = _model_inputs(crash_at=200, crash_depth=0.03)         # 3% crash wipes out >=33x longs opened earlier
    res = estimate_clusters(c, oi, ratio, PublicModelConfig(), 0.005, 15)
    base = estimate_clusters(*_model_inputs(), PublicModelConfig(), 0.005, 15)
    crashed_longs = sum(cl.notional_usd for cl in res.clusters if cl.side == "long")
    steady_longs = sum(cl.notional_usd for cl in base.clusters if cl.side == "long")
    assert res.info["surviving_usd"] < res.info["added_usd"] * 0.9
    # 50x/100x longs (liq within 3% below entry) are gone after the crash; 5x-25x longs remain
    assert not any(cl.side == "long" and cl.price > 97.5 and cl.price < 100 * 0.995 * 0.99 for cl in res.clusters if cl.price > 99.5)
    assert crashed_longs > 0 and steady_longs > 0


def test_model_never_invents_data():
    c, oi, ratio = _model_inputs()
    cfg = PublicModelConfig()
    assert estimate_clusters(c, None, ratio, cfg, 0.005, 15).clusters == []
    r = estimate_clusters(c, oi, None, cfg, 0.005, 15)
    assert r.clusters == [] and "ratio" in r.reason                      # cannot split longs/shorts -> no guess
    assert estimate_clusters(c.iloc[:20], oi, ratio, cfg, 0.005, 15).clusters == []
    sparse = oi.iloc[::10]                                               # OI covers only 10% of bars
    assert "covers only" in estimate_clusters(c, sparse, ratio, cfg, 0.005, 15).reason
    flat_oi = oi.assign(oi=1000.0)
    assert estimate_clusters(c, flat_oi, ratio, cfg, 0.005, 15).clusters == []      # no new exposure -> nothing modelled


class ModelConnector(UniverseConnector):
    """Serves 15m OI/ratio histories aligned with the candle frames (what the public API provides)."""

    async def open_interest_history(self, symbol, period="5m", limit=100, end_ms=None):
        c = tf_candles(self.frames[symbol], 500)["15m"]
        n = len(c)
        return pd.DataFrame({"oi": 1000.0 + 2.0 * np.arange(n)}, index=c.index + pd.Timedelta(minutes=15))

    async def ratio_history(self, kind, symbol, period="5m", limit=100, end_ms=None):
        c = tf_candles(self.frames[symbol], 500)["15m"]
        return pd.DataFrame({"ratio": np.full(len(c), 1.4)}, index=c.index + pd.Timedelta(minutes=15))


def test_public_model_provider_feeds_the_pipeline_without_any_key():
    cfg = default_cfg()
    cfg.liquidation.provider = "public_model"
    cfg.strategy.liquidation_mode = "required"
    bull, _ = build_long_scenario(cfg)
    conn = ModelConnector({"UPAUSDT": bull})
    prov = PublicModelProvider(conn, cfg.liquidation.model, 500, RatioKind.GLOBAL_ACCOUNT, 0.005)
    ti = TradeIntelligence(cfg, Secrets(), conn, prov, Database(":memory:"))
    assert Secrets().liq_api_key is None                                    # no key anywhere
    ti.collector.clock = lambda: bull["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    row = asyncio.run(ti.scan_symbol("UPAUSDT"))
    snap = ti.snapshots["UPAUSDT"]
    assert snap.liquidation_clusters and snap.liquidation_source == "public_model"
    assert "liquidation" not in snap.errors
    assert any("public-data MODEL" in f for f in row.ev.reduced_features)   # always labelled lower-confidence
    assert row.ev.outcome in (Outcome.TRADE_SETUP, Outcome.WATCH, Outcome.NO_TRADE)
    assert not any("required liquidation context missing" in r for r in row.ev.rejections)   # heatmap WAS available


def test_model_failure_is_reported_not_papered_over():
    cfg = default_cfg()
    cfg.strategy.liquidation_mode = "required"
    bull, _ = build_long_scenario(cfg)

    class NoRatio(ModelConnector):
        async def ratio_history(self, kind, symbol, period="5m", limit=100, end_ms=None):
            from src.exchanges.base import NotSupported
            raise NotSupported("no ratio")

    conn = NoRatio({"UPAUSDT": bull})
    ti = TradeIntelligence(cfg, Secrets(), conn, PublicModelProvider(conn, cfg.liquidation.model, 500, RatioKind.GLOBAL_ACCOUNT, 0.005),
                           Database(":memory:"))
    ti.collector.clock = lambda: bull["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    row = asyncio.run(ti.scan_symbol("UPAUSDT"))
    assert "liquidation" in row.errors and row.ev.outcome is not Outcome.TRADE_SETUP


# ------------------------------------------------------------------ public WebSocket (no key)
def test_ws_feed_all_symbols_mode_collects_actual_liquidations():
    import json
    feed = LiveFeed("wss://fstream.binance.com")
    assert "!markPrice@arr@1s" in feed._stream_url() and "!forceOrder@arr" in feed._stream_url()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    feed.handle(json.dumps([{"e": "markPriceUpdate", "s": "ETHUSDT", "p": "3000.5", "E": now_ms},
                            {"e": "markPriceUpdate", "s": "SOLUSDT", "p": "150.1", "E": now_ms}]))
    feed.handle(json.dumps({"stream": "!forceOrder@arr", "data": {"e": "forceOrder", "E": now_ms, "o": {
        "s": "SOLUSDT", "S": "BUY", "q": "100", "p": "151", "ap": "151.2", "z": "100", "T": now_ms}}}))
    assert feed.mark["ETHUSDT"][0] == 3000.5 and len(feed.liquidations) == 1
    ev = feed.recent("SOLUSDT")[0]
    assert ev.side == "short" and feed.recent("ETHUSDT") == []
