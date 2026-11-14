import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from config import Secrets
from src.backtesting.data import fetch_history_backwards, fetch_klines_paged
from src.data_feeds.orderbook import orderbook_metrics
from src.data_feeds.ws_feed import LiveFeed, parse_force_order
from src.database.db import Database
from src.exchanges.binance import parse_klines, parse_symbol_rules
from src.liquidation.heatmap import analyze_clusters, heatmap_support, unavailable
from src.liquidation.providers import FileHeatmapProvider, NullProvider, parse_clusters
from src.models import Direction, LiquidationCluster, OrderBook, RatioKind
from src.pipeline import TradeIntelligence
from src.ui.runtime import BackgroundRunner
from src.ui.tables import metric_freshness, rows_to_frame
from tests.helpers import *

NOW = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
FM = {"items": "data", "price": "price", "notional": "notional_usd", "side": "side", "timestamp": "timestamp"}


def test_parse_klines_drops_the_forming_candle():
    t0 = int(datetime(2025, 3, 1, 11, 50, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [[t0 + i * 300000, "1", "2", "0.5", "1.5", "10", t0 + i * 300000 + 299999, "0", 1, "0", "0", "0"] for i in range(4)]
    df = parse_klines(rows, now=NOW)                       # bars at 11:50, 11:55, 12:00, 12:05
    assert list(df.index.strftime("%H:%M")) == ["11:50", "11:55"]       # 12:00 candle is still forming at 12:00:00
    assert df["close"].dtype == float and df.index.tz is not None


def test_parse_symbol_rules_and_missing_symbol():
    info = {"symbols": [{"symbol": "BTCUSDT", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.10"}, {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
        {"filterType": "MIN_NOTIONAL", "notional": "100"}]}]}
    r = parse_symbol_rules(info, "BTCUSDT")
    assert r.tick_size == Decimal("0.10") and r.step_size == Decimal("0.001") and r.min_notional == Decimal("100")
    try:
        parse_symbol_rules(info, "NOPEUSDT")
        raise AssertionError
    except Exception as e:
        assert "not found" in str(e)


def test_force_order_parsing_separates_actual_events():
    msg = {"e": "forceOrder", "E": 1700000000000, "o": {"s": "BTCUSDT", "S": "SELL", "q": "0.5", "p": "60000", "ap": "59950", "z": "0.5", "T": 1700000000000}}
    ev = parse_force_order(msg)
    assert ev.side == "long" and ev.price == 59950 and abs(ev.notional - 29975) < 1e-6
    assert parse_force_order({"e": "forceOrder", "o": {"S": "BUY"}}) is None and parse_force_order({"e": "x"}) is None
    feed = LiveFeed("wss://x", ["BTCUSDT"])
    feed.handle(json.dumps({"stream": "!forceOrder@arr", "data": msg}))
    feed.handle(json.dumps({"data": {"e": "markPriceUpdate", "s": "BTCUSDT", "p": "60001.5", "E": 1700000001000}}))
    assert len(feed.liquidations) == 1 and feed.mark["BTCUSDT"][0] == 60001.5


def test_file_provider_never_invents_data():
    assert asyncio.run(NullProvider().fetch_clusters("BTCUSDT")) == []
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "{symbol}.json"
        prov = FileHeatmapProvider(str(p), FM)
        assert asyncio.run(prov.fetch_clusters("BTCUSDT")) == []                       # no file -> nothing
        (Path(d) / "BTCUSDT.json").write_text(json.dumps({"data": [
            {"price": 61000, "notional_usd": 5e8, "side": "short", "timestamp": "2025-03-01T11:59:00Z"},
            {"price": "bad", "notional_usd": 1}, {"price": -5, "notional_usd": 1}, {"price": 59000, "notional_usd": float("nan")}]}))
        cl = asyncio.run(prov.fetch_clusters("BTCUSDT"))
        assert len(cl) == 1 and cl[0].estimated and cl[0].timestamp is not None


def test_heatmap_ranking_sweep_and_staleness():
    cfg = default_cfg()
    idx = pd.date_range(NOW - timedelta(hours=2), periods=24, freq="5min", tz="UTC")
    c5 = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}, index=idx)
    c5["close_time"] = idx + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
    t = NOW - timedelta(minutes=30)
    clusters = [LiquidationCluster(101.0, 4e8, "short", t, "x"), LiquidationCluster(103.5, 9e8, "short", t, "x"),
                LiquidationCluster(99.0, 2e8, "long", t, "x"), LiquidationCluster(100.2, 5e8, "short", t, "x")]
    ctx = analyze_clusters(clusters, 100.0, c5, NOW, cfg.strategy)
    by_price = {c.cluster.price: c for c in ctx.clusters}
    assert by_price[100.2].swept is True and by_price[100.2].rank == 0            # high of 100.5 traded through 100.2
    assert by_price[101.0].swept is False and abs(by_price[101.0].distance_pct - 1.0) < 1e-9
    assert ctx.nearest_above.cluster.price == 101.0 and ctx.nearest_below.cluster.price == 99.0
    assert by_price[103.5].rank < by_price[101.0].rank * 2.5                         # distance decay limits far clusters
    score, ok, why = heatmap_support(Direction.LONG, ctx)
    assert score is not None and "not guaranteed" in why
    stale = analyze_clusters(clusters, 100.0, c5, NOW + timedelta(hours=5), cfg.strategy)
    assert not stale.available and "stale" in stale.reason
    assert heatmap_support(Direction.SHORT, stale)[0] is None and heatmap_support(Direction.LONG, None)[0] is None
    assert not analyze_clusters([], 100.0, c5, NOW, cfg.strategy).available
    notime = analyze_clusters([LiquidationCluster(101, 1e8, "short", None, "x")], 100.0, c5, NOW, cfg.strategy)
    assert not notime.available


def test_orderbook_metrics():
    book = OrderBook([(99.9, 100), (99.5, 50)], [(100.1, 20), (100.4, 10)], None, "t")
    m = orderbook_metrics(book, 0.5)
    assert abs(m.spread_bps - 20.0) < 0.1 and m.bid_notional > m.ask_notional and 0 < m.imbalance < 1
    assert orderbook_metrics(OrderBook([], [], None, "t")) is None
    assert orderbook_metrics(OrderBook([(101, 1)], [(100, 1)], None, "t")) is None            # crossed book = bad data


def test_history_paging_has_no_gaps_or_duplicates():
    full = base_uptrend(6000, seed=5)
    full = full.assign(close_time=full["close_time"])

    class Conn:
        calls = 0

        async def klines(self, symbol, interval, limit=500, start_ms=None, end_ms=None):
            self.calls += 1
            s = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            return full.loc[s:].head(limit)

    end = full.index[-1] + pd.Timedelta(minutes=5)
    conn = Conn()
    got = asyncio.run(fetch_klines_paged(conn, "X", "5m", days=(end - full.index[0]).total_seconds() / 86400, page=1500, now=end))
    assert conn.calls >= 4 and got.index.is_unique and got.index.is_monotonic_increasing and len(got) == len(full)

    ratio = pd.DataFrame({"ratio": range(3000)}, index=pd.date_range(end - pd.Timedelta(days=11), periods=3000, freq="5min", tz="UTC"))

    async def fetch(limit, end_ms):
        e = pd.Timestamp(end_ms, unit="ms", tz="UTC")
        return ratio.loc[:e].tail(limit)

    out = asyncio.run(fetch_history_backwards(fetch, days=10, now=ratio.index[-1]))
    assert out.index.is_unique and out.index.is_monotonic_increasing and len(out) > 2800


def test_dashboard_tables_show_sources_freshness_and_rejections():
    cfg = default_cfg()
    df, _ = build_short_scenario(cfg)
    now = df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)

    ti = TradeIntelligence(cfg, Secrets(), FakeConnector(df, ratio=0.85), NullProvider(), Database(":memory:"))
    ti.collector.clock = lambda: now
    row = asyncio.run(ti.scan_symbol("TESTUSDT"))
    frame = rows_to_frame([row])
    assert {"Symbol", "Price", "Outcome", "Score", "Entry", "SL", "TP1", "TP2", "TP3", "R/R (net)", "Rejection reason"} <= set(frame.columns)
    assert frame.iloc[0]["Outcome"] != "TRADE SETUP"
    assert "SHORT blocked" in frame.iloc[0]["Rejection reason"] and "0.850" in frame.iloc[0]["Rejection reason"]
    sig = ti.db.query("SELECT * FROM signals")[0]
    assert "SHORT blocked" in json.loads(json.dumps(sig["payload"]))                 # also stored for the journal / logs
    fresh = metric_freshness(ti.snapshots["TESTUSDT"], cfg, now)
    assert {"Metric", "Source", "Timestamp", "Age (s)", "Status"} <= set(fresh.columns)
    assert "Liquidation clusters (ESTIMATED)" in set(fresh["Metric"]) and (fresh["Status"] == "MISSING").any()
    assert fresh.loc[fresh["Metric"].str.startswith("L/S ratio"), "Source"].iloc[0].endswith("global_account")


def test_background_runner_scans_without_blocking_and_survives_errors():
    cfg = default_cfg()
    cfg.symbols, cfg.scan_interval_seconds = ["TESTUSDT"], 10
    df, _ = build_long_scenario(cfg)
    ti = TradeIntelligence(cfg, Secrets(), FakeConnector(df, ratio=0.9), NullProvider(), Database(":memory:"))
    ti.collector.clock = lambda: df["close_time"].iloc[-1].to_pydatetime() + timedelta(seconds=2)
    runner = BackgroundRunner(ti).start()
    assert runner.wait_for_first_scan(20) and "TESTUSDT" in runner.rows and runner.last_error is None
