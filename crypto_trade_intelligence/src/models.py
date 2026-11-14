"""Core data models shared across modules. Every metric carries source + timestamp."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import pandas as pd


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1

    @property
    def opposite(self) -> "Direction":
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


class Outcome(str, Enum):
    TRADE_SETUP = "TRADE SETUP"
    WATCH = "WATCH / WAIT FOR CONFIRMATION"
    NO_TRADE = "NO TRADE"


class TrendLabel(str, Enum):
    STRONG_UP = "Strong uptrend"
    WEAK_UP = "Weak uptrend"
    RANGE = "Range-bound"
    WEAK_DOWN = "Weak downtrend"
    STRONG_DOWN = "Strong downtrend"
    UNCERTAIN = "Uncertain / conflicting"

    @property
    def direction(self) -> int:
        return {"Strong uptrend": 1, "Weak uptrend": 1, "Weak downtrend": -1, "Strong downtrend": -1}.get(self.value, 0)

    @property
    def is_strong(self) -> bool:
        return self in (TrendLabel.STRONG_UP, TrendLabel.STRONG_DOWN)


class RatioKind(str, Enum):
    GLOBAL_ACCOUNT = "global_account"
    TOP_TRADER_ACCOUNT = "top_trader_account"
    TOP_TRADER_POSITION = "top_trader_position"


@dataclass(frozen=True)
class RatioReading:
    """A long/short ratio value. `kind` is carried so metrics are never mixed."""
    value: float | None
    kind: RatioKind
    timestamp: datetime | None
    source: str
    period: str = ""


@dataclass(frozen=True)
class TimedValue:
    value: float | None
    source: str
    timestamp: datetime | None


@dataclass
class OrderBook:
    bids: list[tuple[float, float]]   # (price, qty) best first
    asks: list[tuple[float, float]]
    timestamp: datetime | None
    source: str


@dataclass(frozen=True)
class OrderBookMetrics:
    mid: float
    spread_bps: float
    bid_notional: float
    ask_notional: float
    imbalance: float          # (bid-ask)/(bid+ask) within depth window, -1..1
    timestamp: datetime | None


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal
    max_qty: Decimal | None = None
    source: str = "exchange"


@dataclass(frozen=True)
class LiquidationCluster:
    """ESTIMATED liquidation concentration from a heatmap provider (not actual events)."""
    price: float
    notional_usd: float
    side: str                 # "long" (long liquidations, normally below price) | "short" | "unknown"
    timestamp: datetime | None
    source: str
    estimated: bool = True


@dataclass(frozen=True)
class LiquidationEvent:
    """An ACTUAL liquidation order observed on the public exchange feed."""
    symbol: str
    side: str                 # "long" or "short" position that was liquidated
    price: float
    qty: float
    timestamp: datetime
    source: str

    @property
    def notional(self) -> float:
        return self.price * self.qty


@dataclass
class MarketSnapshot:
    symbol: str
    now: datetime
    price: float | None = None
    mark_price: float | None = None
    volume_24h_usd: float | None = None
    candles: dict[str, pd.DataFrame] = field(default_factory=dict)
    funding: TimedValue | None = None
    funding_history: pd.DataFrame | None = None
    oi_history: pd.DataFrame | None = None
    ratios: dict[RatioKind, RatioReading] = field(default_factory=dict)
    ratio_history: dict[RatioKind, pd.DataFrame] = field(default_factory=dict)
    orderbook: OrderBook | None = None
    liquidation_clusters: list[LiquidationCluster] | None = None
    liquidation_source: str = "none"
    liquidation_events: list[LiquidationEvent] = field(default_factory=list)
    rules: SymbolRules | None = None
    errors: dict[str, str] = field(default_factory=dict)   # metric -> why it is missing
    timestamps: dict[str, datetime | None] = field(default_factory=dict)


def jsonable(obj: Any) -> Any:
    """Best-effort conversion for storing dataclasses/enums/Decimals in SQLite JSON columns."""
    import dataclasses
    import math
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame {obj.shape}>"
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return jsonable(obj.item())
    except ImportError:  # pragma: no cover
        pass
    return obj
