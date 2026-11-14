"""Configuration (pydantic-validated). Secrets come from environment variables only."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent

# The mandatory SHORT filter threshold. Deliberately NOT a config option.
SHORT_MIN_RATIO = 1.0

RatioKindName = Literal["global_account", "top_trader_account", "top_trader_position"]


class ExchangeConfig(BaseModel):
    name: Literal["binance_usdm", "ccxt"] = "binance_usdm"
    ccxt_id: str = "bybit"                       # only used when name == "ccxt"
    base_url: str = "https://fapi.binance.com"
    ws_url: str = "wss://fstream.binance.com"
    requests_per_second: float = Field(8.0, gt=0, le=40)
    weight_per_minute: int = Field(1800, gt=0)          # Binance allows 2400/min/IP; stay well under
    data_requests_per_5min: int = Field(900, gt=0)      # /futures/data/* (ratios, OI history): ~1000 per 5 min per IP
    max_retries: int = Field(5, ge=0, le=10)
    timeout_s: float = Field(10.0, gt=0)
    live_feed: bool = True                              # public WebSocket: mark prices + ACTUAL liquidation events


class RatioConfig(BaseModel):
    kind: RatioKindName = "global_account"       # never mixed with the other kinds
    period: Literal["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"] = "5m"
    max_age_seconds: int = Field(1200, gt=0)     # older than this == stale == SHORT prohibited


class PublicModelConfig(BaseModel):
    """Assumptions of the keyless, self-computed liquidation estimate (see src/liquidation/model.py)."""
    period: Literal["5m", "15m", "30m", "1h"] = "15m"
    bucket_pct: float = Field(0.25, gt=0)                # price-bucket width as % of price
    leverage_mix: dict = Field(default_factory=lambda: {5: 0.10, 10: 0.30, 25: 0.30, 50: 0.20, 100: 0.10})
    min_cluster_share: float = Field(0.03, ge=0, le=1)   # drop clusters smaller than this share of the largest
    max_clusters: int = Field(40, ge=1)
    min_coverage: float = Field(0.8, gt=0, le=1)         # share of bars that must have OI AND ratio data

    @model_validator(mode="after")
    def _check(self):
        if not self.leverage_mix or sum(self.leverage_mix.values()) <= 0:
            raise ValueError("leverage_mix must have positive weights")
        return self


class LiquidationConfig(BaseModel):
    # public_model: NO KEY. Estimated from public Binance OI + long/short ratio + leverage-mix assumption (lower confidence).
    # file / http_json: licensed provider data (e.g. CoinGlass). none: no heatmap.
    provider: Literal["none", "public_model", "file", "http_json"] = "public_model"
    model: PublicModelConfig = Field(default_factory=PublicModelConfig)
    # file provider: path template, e.g. data/heatmaps/{symbol}.json
    path_template: str = "data/heatmaps/{symbol}.json"
    # http_json provider: licensed provider URL template + header names; key from env LIQ_API_KEY
    url_template: str = ""
    api_key_header: str = "X-API-KEY"
    field_map: dict = Field(default_factory=lambda: {
        "items": "data", "price": "price", "notional": "notional_usd",
        "side": "side", "timestamp": "timestamp"})
    ttl_seconds: int = Field(120, ge=1)


class StrategyConfig(BaseModel):
    candle_limit: int = Field(500, ge=250, le=1500)
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    adx_period: int = 14
    adx_strong: float = 25.0
    adx_range: float = 18.0
    swing_left: int = Field(3, ge=1)
    swing_right: int = Field(3, ge=1)
    event_lookback: int = 12                     # bars in which a BOS/CHoCH counts as "recent"
    min_score: float = Field(65.0, ge=0, le=100)
    min_rr: float = Field(2.0, gt=0)             # net of estimated costs
    proximity_atr: float = 0.6
    sweep_lookback_bars: int = 12                # 5m bars
    retest_lookback_bars: int = 16               # 15m bars
    stop_atr_buffer: float = 0.3
    min_stop_atr: float = 0.7
    max_stop_atr: float = 3.5
    max_stop_pct: float = 5.0
    tp_buffer_atr: float = 0.05
    min_tp1_r: float = 1.0
    obstacle_min_touches: int = 3                # a level with >= this many touches blocks a nearer TP
    tp_split: list = Field(default_factory=lambda: [0.5, 0.3, 0.2])
    min_volume_ratio: float = 1.1
    volume_ma: int = 20
    reversal_lookback_bars: int = 12             # 1h bars in which a CHoCH confirms a reversal
    max_candle_lag_bars: int = 3
    oi_window_minutes: int = 60
    oi_contradiction_pct: float = 1.0
    funding_extreme: float = 0.0005
    ratio_crowded: float = 2.0
    liquidation_mode: Literal["required", "optional"] = "required"
    heatmap_min_support: float = 0.5
    heatmap_max_distance_pct: float = 5.0
    heatmap_decay_pct: float = 2.0
    heatmap_max_age_seconds: int = 3600
    require_orderbook: bool = True
    orderbook_depth_pct: float = 0.5
    max_spread_bps: float = 8.0
    min_depth_notional_usd: float = 100_000.0
    min_volume_24h_usd: float = 20_000_000.0         # single source for universe filter AND engine check (0 = no floor)
    expected_hold_hours: float = 8.0
    weights: dict = Field(default_factory=lambda: {
        "htf_trend": 20, "structure": 15, "liquidation": 15, "volume": 10, "oi": 8,
        "ratio": 7, "funding": 5, "orderbook": 5, "rr": 10, "data_quality": 5})

    @model_validator(mode="after")
    def _check(self):
        if self.min_stop_atr >= self.max_stop_atr:
            raise ValueError("min_stop_atr must be < max_stop_atr")
        if not self.tp_split or abs(sum(self.tp_split) - 1.0) > 1e-6:
            raise ValueError("tp_split must sum to 1.0")
        if sum(self.weights.values()) <= 0:
            raise ValueError("weights must sum to > 0")
        return self


class RiskConfig(BaseModel):
    account_equity: float = Field(10_000.0, gt=0)
    risk_per_trade_pct: float = Field(0.5, gt=0, le=5)
    leverage: float = Field(5.0, ge=1, le=125)
    margin_allocation_pct: float = Field(20.0, gt=0, le=100)   # max margin one position may use
    max_daily_loss_pct: float = Field(2.0, gt=0, le=100)
    max_positions: int = Field(3, ge=1)
    max_total_exposure_pct: float = Field(300.0, gt=0)         # total notional / equity
    max_slippage_pct: float = Field(0.05, ge=0)
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    maintenance_margin_rate: float = 0.005
    liquidation_safety: float = 0.8               # stop must be inside this share of the liq distance
    breakeven_after_tp1: bool = False
    max_hold_hours: float = 24.0


class ExecutionConfig(BaseModel):
    # analysis (default) -> only signals; paper -> simulated fills; live -> real orders (needs extra gates)
    mode: Literal["analysis", "paper", "live"] = "analysis"
    exchange_testnet: bool = True


class AlertConfig(BaseModel):
    telegram_enabled: bool = False
    cooldown_minutes: int = Field(60, ge=0)
    min_score_change: float = 10.0
    entry_shift_atr: float = 0.5


class StorageConfig(BaseModel):
    db_path: str = "data/trade_intel.sqlite3"
    log_dir: str = "logs"
    log_level: str = "INFO"


class UniverseConfig(BaseModel):
    mode: Literal["all", "fixed"] = "all"                # all: every TRADING USDT perpetual; fixed: use `symbols`
    max_symbols: int = Field(200, ge=1)                  # after the volume filter, highest 24h volume first
    min_listing_days: int = Field(45, ge=0)              # 4h EMA200 needs ~35 days of candles
    exclude: list = Field(default_factory=lambda: ["USDCUSDT"])
    triage_refresh_seconds: int = Field(900, ge=30)      # 1h/4h bias per coin is re-checked at most this often
    max_deep_scans_per_cycle: int = Field(30, ge=1)      # full 5m..4h + positioning scans per cycle (rotating)
    concurrency: int = Field(4, ge=1, le=16)


class AppConfig(BaseModel):
    symbols: list = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    scan_interval_seconds: int = Field(60, ge=10)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    ratio: RatioConfig = Field(default_factory=RatioConfig)
    liquidation: LiquidationConfig = Field(default_factory=LiquidationConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


@dataclass(frozen=True)
class Secrets:
    """Environment-only secrets. repr() never shows values."""
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    liq_api_key: str | None = None
    allow_live_orders: bool = False

    @classmethod
    def from_env(cls) -> "Secrets":
        g = os.environ.get
        return cls(g("BINANCE_API_KEY"), g("BINANCE_API_SECRET"), g("TELEGRAM_BOT_TOKEN"),
                   g("TELEGRAM_CHAT_ID"), g("LIQ_API_KEY"),
                   g("ALLOW_LIVE_ORDERS", "").strip().lower() == "yes-i-accept-the-risk")

    def __repr__(self) -> str:  # pragma: no cover
        present = {k: bool(v) for k, v in self.__dict__.items()}
        return f"Secrets({present})"


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    """Load .env (if python-dotenv is installed) and an optional YAML file, then validate."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:  # pragma: no cover
        pass
    p = Path(path) if path else ROOT / "config.yaml"
    if p.exists():
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return AppConfig.model_validate(data)
    return AppConfig()
