"""Order-book depth, bid/ask imbalance and spread."""
from __future__ import annotations

from src.models import OrderBook, OrderBookMetrics
from src.utils import is_finite


def orderbook_metrics(book: OrderBook | None, depth_pct: float = 0.5) -> OrderBookMetrics | None:
    if book is None or not book.bids or not book.asks:
        return None
    bid, ask = book.bids[0][0], book.asks[0][0]
    if not (is_finite(bid) and is_finite(ask)) or bid <= 0 or ask <= 0 or bid >= ask:
        return None
    mid = (bid + ask) / 2
    lo, hi = mid * (1 - depth_pct / 100), mid * (1 + depth_pct / 100)
    bn = sum(p * q for p, q in book.bids if p >= lo)
    an = sum(p * q for p, q in book.asks if p <= hi)
    tot = bn + an
    if tot <= 0:
        return None
    return OrderBookMetrics(mid, (ask - bid) / mid * 1e4, bn, an, (bn - an) / tot, book.timestamp)
