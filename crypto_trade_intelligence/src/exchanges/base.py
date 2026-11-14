from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.models import OrderBook, RatioKind, SymbolRules, TimedValue


class NotSupported(RuntimeError):
    """The exchange does not expose this metric. The caller records it as missing; nothing is inferred."""


class ExchangeConnector(ABC):
    name: str = "base"

    @abstractmethod
    async def klines(self, symbol: str, interval: str, limit: int = 500, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame: ...
    @abstractmethod
    async def ticker(self, symbol: str) -> tuple[float, float, "pd.Timestamp"]: ...     # last, 24h quote volume, time
    @abstractmethod
    async def mark_and_funding(self, symbol: str) -> tuple[float | None, TimedValue]: ...
    @abstractmethod
    async def funding_history(self, symbol: str, limit: int = 100) -> pd.DataFrame: ...
    @abstractmethod
    async def open_interest_history(self, symbol: str, period: str = "5m", limit: int = 100) -> pd.DataFrame: ...
    @abstractmethod
    async def ratio_history(self, kind: RatioKind, symbol: str, period: str = "5m", limit: int = 100) -> pd.DataFrame: ...
    @abstractmethod
    async def order_book(self, symbol: str, limit: int = 100) -> OrderBook: ...
    @abstractmethod
    async def symbol_rules(self, symbol: str) -> SymbolRules: ...

    async def universe(self) -> list[dict]:
        """All tradable USDT perpetuals: [{'symbol': 'BTCUSDT', 'onboard': datetime|None}, ...]."""
        raise NotSupported("universe discovery not supported by this connector")

    async def all_tickers(self) -> pd.DataFrame:
        """DataFrame indexed by symbol with columns last, quote_volume (USDT, 24h), change_pct."""
        raise NotSupported("bulk tickers not supported by this connector")
