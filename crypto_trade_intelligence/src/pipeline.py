"""Orchestration: collect -> evaluate -> record -> alert -> (paper | live) execute."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import AppConfig, Secrets
from src.alerts.telegram import AlertManager, ConsoleNotifier, TelegramNotifier
from src.data_feeds.builder import build_input
from src.data_feeds.collector import MarketDataCollector
from src.database.db import Database
from src.exchanges.binance import BinanceUSDM
from src.exchanges.ccxt_connector import CcxtConnector
from src.exchanges.http import HttpClient
from src.execution.guards import pre_trade_check
from src.execution.paper import PaperBroker
from src.data_feeds.ws_feed import LiveFeed
from src.liquidation.model import PublicModelProvider
from src.liquidation.providers import FileHeatmapProvider, HttpJsonProvider, NullProvider
from src.models import Outcome, RatioKind
from src.positioning.ratio import latest_reading
from src.risk.manager import RiskManager
from src.risk.sizing import SizingResult, size_position
from src.strategies.engine import Evaluation, StrategyEngine
from src.universe import TriageResult, UniverseStats, prioritize, select_candidates, triage_symbol
from src.utils import log_event, utcnow

log = logging.getLogger("pipeline")


@dataclass
class ScanRow:
    symbol: str
    ev: Evaluation
    sizing: SizingResult | None = None
    action: str = "none"
    notes: list[str] = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    signal_id: int | None = None


def day_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


class TradeIntelligence:
    def __init__(self, cfg: AppConfig, secrets: Secrets, connector, provider, db: Database, notifier=None, http=None, feed=None):
        self.cfg, self.secrets, self.db, self.feed = cfg, secrets, db, feed
        self.http = http
        mode = "live" if cfg.execution.mode == "live" else "paper"
        self.risk = RiskManager(cfg.risk, lambda now: db.realized_since(mode, day_start(now)), lambda: db.realized_total(mode))
        self.collector = MarketDataCollector(connector, provider, cfg, feed=feed)
        self.connector = connector
        self.engine = StrategyEngine(cfg.strategy, cfg.risk, cfg.ratio)
        self.notifier = notifier or ConsoleNotifier()
        self.alerts = AlertManager(cfg, db, self.notifier)
        self.paper = PaperBroker(cfg, db, self.risk) if cfg.execution.mode in ("paper", "analysis") else None
        self.live = None
        if cfg.execution.mode == "live":
            from src.execution.live import LiveBroker
            self.live = LiveBroker(cfg, secrets, self.risk)      # raises unless every live gate is satisfied
        self._prev_clusters: dict[str, list] = {}
        self.snapshots: dict = {}                      # last MarketSnapshot per symbol (read by the UI)
        self.triage: dict[str, TriageResult] = {}      # last 1h/4h triage per coin (all-coins mode)
        self.universe_stats = UniverseStats()
        self._deep_cursor = 0
        self._last_sig: dict[str, tuple] = {}
        self.last_error: str | None = None

    @classmethod
    def from_config(cls, cfg: AppConfig, secrets: Secrets | None = None, db: Database | None = None) -> "TradeIntelligence":
        secrets = secrets or Secrets.from_env()
        ex = cfg.exchange
        http = HttpClient(ex.base_url, ex.requests_per_second, ex.max_retries, ex.timeout_s,
                          weight_per_minute=ex.weight_per_minute, data_per_5min=ex.data_requests_per_5min)
        connector = BinanceUSDM(http) if ex.name == "binance_usdm" else CcxtConnector(ex.ccxt_id)
        lc = cfg.liquidation
        if lc.provider == "file":
            provider = FileHeatmapProvider(lc.path_template, lc.field_map)
        elif lc.provider == "http_json":
            provider = HttpJsonProvider(http, lc.url_template, secrets.liq_api_key, lc.api_key_header, lc.field_map, lc.ttl_seconds)
        elif lc.provider == "public_model":                   # keyless: estimate from public OI + ratio + leverage-mix assumption
            provider = PublicModelProvider(connector, lc.model, cfg.strategy.candle_limit, RatioKind(cfg.ratio.kind),
                                           cfg.risk.maintenance_margin_rate)
        else:
            provider = NullProvider()
        feed = None
        if ex.live_feed and ex.name == "binance_usdm":
            feed = LiveFeed(ex.ws_url, None if cfg.universe.mode == "all" else cfg.symbols)
        notifier = None
        if cfg.alerts.telegram_enabled:
            if not (secrets.telegram_bot_token and secrets.telegram_chat_id):
                raise RuntimeError("alerts.telegram_enabled=true but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set")
            notifier = TelegramNotifier(secrets.telegram_bot_token, secrets.telegram_chat_id)
        return cls(cfg, secrets, connector, provider, db or Database(cfg.storage.db_path), notifier, http, feed)

    # ----------------------------------------------------------------
    async def scan_symbol(self, symbol: str) -> ScanRow:
        snap = await self.collector.snapshot(symbol)
        self.snapshots[symbol] = snap
        inp = build_input(snap, self.cfg, self._prev_clusters.get(symbol))
        if snap.liquidation_clusters:
            self._prev_clusters[symbol] = snap.liquidation_clusters
        ev = self.engine.evaluate(inp)
        accepted = ev.outcome is Outcome.TRADE_SETUP
        row = ScanRow(symbol, ev, errors=snap.errors)
        fp = (ev.outcome.value, ev.direction.value if ev.direction else None, tuple((ev.rejections + ev.other_notes)[:2]))
        if accepted or self._last_sig.get(symbol) != fp:        # journal state CHANGES, not an identical line every minute
            row.signal_id = self.db.record_signal(ev, accepted)
            self._last_sig[symbol] = fp
        if self.paper and snap.candles.get("5m") is not None:
            self.paper.update(symbol, snap.candles["5m"], snap.funding.value if snap.funding else None)
        if not accepted:
            return row
        r = self.cfg.risk
        exposure_left = self.risk.remaining_exposure(self.paper.open_notionals() if self.paper else [])
        row.sizing = size_position(ev.plan.entry, ev.plan.stop_loss, self.risk.equity, r.risk_per_trade_pct, r.leverage,
                                   snap.rules, r.margin_allocation_pct, r.taker_fee, r.max_slippage_pct, exposure_left)
        tick = snap.rules.tick_size if snap.rules else None
        sent, why = await self.alerts.maybe_alert(ev, snap.now, tick)
        row.notes.append(f"alert: {why}")
        mode = self.cfg.execution.mode
        if mode == "analysis":
            row.action = "alerted" if sent else "analysis-only"
        elif mode == "paper":
            tid, reasons = self.paper.submit(ev, row.sizing, snap.now, row.signal_id, None, snap.price)
            row.action = f"paper_opened#{tid}" if tid else "paper_rejected"
            row.notes += reasons
        elif mode == "live":
            row.action, extra = await self._live(ev, row, snap)
            row.notes += extra
        log_event(log, logging.INFO, "scan_result", symbol=symbol, action=row.action, notes=row.notes)
        return row

    async def _live(self, ev, row: ScanRow, snap) -> tuple[str, list[str]]:
        fresh = None
        if ev.direction.value == "SHORT":                 # re-fetch the ratio: gate 3 of 3 uses live data, not the cached signal
            kind = RatioKind(self.cfg.ratio.kind)
            try:
                df = await self.connector.ratio_history(kind, ev.symbol, self.cfg.ratio.period, 3)
                fresh = latest_reading(df, kind, "live-refetch", self.cfg.ratio.period)
            except Exception as e:
                return "live_blocked", [f"could not refresh ratio: {type(e).__name__}"]
        try:
            orders = await self.live.place_bracket(ev, row.sizing, utcnow(), self.paper.open_notionals() if self.paper else [], fresh)
            return "live_placed", [str(list(orders))]
        except Exception as e:
            return "live_blocked", [str(e)[:200]]

    async def _bounded(self, symbols: list[str]) -> list[ScanRow]:
        sem = asyncio.Semaphore(self.cfg.universe.concurrency)

        async def one(sym):
            async with sem:
                return await self.scan_symbol(sym)

        rows = await asyncio.gather(*(one(s) for s in symbols), return_exceptions=True)
        out = []
        for s, r in zip(symbols, rows):
            if isinstance(r, BaseException):
                log_event(log, logging.ERROR, "scan_failed", symbol=s, error=f"{type(r).__name__}: {str(r)[:200]}")
            else:
                out.append(r)
        return out

    async def scan_universe(self) -> list[ScanRow]:
        """All-coins mode: discover -> filter -> triage (1h/4h) -> deep-scan the coins that matter."""
        uc, now = self.cfg.universe, utcnow()
        uni, tick = await asyncio.gather(self.connector.universe(), self.connector.all_tickers())
        symbols, skipped = select_candidates(uni, tick, self.cfg, now)
        stale = [s for s in symbols if s not in self.triage or (now - self.triage[s].at).total_seconds() > uc.triage_refresh_seconds]
        sem = asyncio.Semaphore(uc.concurrency)

        async def tri(sym):
            async with sem:
                return await triage_symbol(self.connector, sym, self.cfg, now)

        for r in await asyncio.gather(*(tri(s) for s in stale)):
            self.triage[r.symbol] = r
        passed = [s for s in symbols if self.triage[s].passed]
        held = sorted({p.symbol for p in self.paper.positions.values()}) if self.paper else []
        interesting = [s for s, sig in self._last_sig.items() if sig[0] != Outcome.NO_TRADE.value and s in symbols]
        deep, self._deep_cursor = prioritize(passed, held + [s for s in interesting if s not in held], uc.max_deep_scans_per_cycle, self._deep_cursor)
        rows = await self._bounded(deep)
        for r in self.triage.values():
            if not r.passed:
                skipped[f"triage: {r.reason.split(':')[0][:40]}"] = skipped.get(f"triage: {r.reason.split(':')[0][:40]}", 0) + (1 if r.symbol in symbols else 0)
        self.universe_stats = UniverseStats(now, len(uni), len(symbols), len(passed), len(symbols) - len(passed), len(rows), skipped)
        log_event(log, logging.INFO, "universe_scan", total=len(uni), candidates=len(symbols), passed_triage=len(passed), deep=len(rows))
        return rows

    async def scan_all(self) -> list[ScanRow]:
        try:
            rows = await (self.scan_universe() if self.cfg.universe.mode == "all" else self._bounded(self.cfg.symbols))
            self.last_error = None
            return rows
        except Exception as e:                                # outage / ban / bad response: report, never guess
            self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            log_event(log, logging.ERROR, "scan_all_failed", error=self.last_error)
            return []

    def start_feed(self):
        """Start the public WebSocket feed (actual liquidation events, mark prices) as a background task."""
        if self.feed is None:
            return None
        return asyncio.get_running_loop().create_task(self.feed.run())

    async def run_forever(self, on_rows=None) -> None:
        self.start_feed()
        while True:
            rows = await self.scan_all()
            if on_rows:
                on_rows(rows)
            await asyncio.sleep(self.cfg.scan_interval_seconds)
