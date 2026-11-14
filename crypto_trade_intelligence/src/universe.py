"""All-coins scanning: discover every USDT perpetual, filter cheaply, triage by 1h/4h bias, then deep-scan.

Funnel (so 300+ coins fit inside the public API's rate limits):
  1. universe + bulk 24h tickers      -> 2 calls total (cached)
  2. volume / listing-age / exclusions -> no calls
  3. triage: 1h + 4h candles per coin  -> ~4 request-weight per coin, refreshed every ~15 min
  4. deep scan: full snapshot          -> ~35 weight per coin, only for coins that passed triage (rotating, capped)
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from config import AppConfig, UniverseConfig
from src.market_structure.trend import analyze_mtf


@dataclass
class UniverseStats:
    timestamp: datetime | None = None
    total: int = 0
    candidates: int = 0
    triaged_pass: int = 0
    triaged_skip: int = 0
    deep_scanned: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


@dataclass
class TriageResult:
    symbol: str
    passed: bool
    bias: str
    labels: dict[str, str]
    reason: str
    at: datetime


def select_candidates(universe: list[dict], tickers: pd.DataFrame, cfg: AppConfig, now: datetime) -> tuple[list[str], dict[str, int]]:
    uc: UniverseConfig = cfg.universe
    skipped: Counter = Counter()
    keep: list[tuple[str, float]] = []
    excluded = {s.upper() for s in uc.exclude}
    for u in universe:
        sym = u["symbol"]
        if sym.upper() in excluded:
            skipped["excluded"] += 1
        elif sym not in tickers.index:
            skipped["no ticker"] += 1
        elif u.get("onboard") is not None and (now - u["onboard"]).days < uc.min_listing_days:
            skipped[f"listed < {uc.min_listing_days} days"] += 1
        elif float(tickers.at[sym, "quote_volume"]) < cfg.strategy.min_volume_24h_usd:
            skipped["24h volume below minimum"] += 1
        else:
            keep.append((sym, float(tickers.at[sym, "quote_volume"])))
    keep.sort(key=lambda x: -x[1])
    if len(keep) > uc.max_symbols:
        skipped[f"beyond max_symbols ({uc.max_symbols})"] += len(keep) - uc.max_symbols
    return [s for s, _ in keep[:uc.max_symbols]], dict(skipped)


async def triage_symbol(connector, symbol: str, cfg: AppConfig, now: datetime) -> TriageResult:
    """Cheap 1h/4h look. Passes coins with a clear higher-timeframe bias or a fresh 1h CHoCH (possible reversal)."""
    try:
        h1, h4 = await asyncio.gather(connector.klines(symbol, "1h", 250), connector.klines(symbol, "4h", 250))
    except Exception as e:                                           # never guess: a coin we cannot read is skipped
        return TriageResult(symbol, False, "unknown", {}, f"data error: {type(e).__name__}", now)
    mtf = analyze_mtf({"1h": h1, "4h": h4}, cfg.strategy)
    labels = {tf: mtf.frames[tf].label.value for tf in ("1h", "4h")}
    if mtf.bias == "unknown":
        return TriageResult(symbol, False, mtf.bias, labels, "; ".join(mtf.notes) or "insufficient history", now)
    choch = mtf.frames["1h"].structure.choch
    if mtf.bias in ("bullish", "bearish"):
        return TriageResult(symbol, True, mtf.bias, labels, "clear higher-timeframe bias", now)
    if choch:
        return TriageResult(symbol, True, mtf.bias, labels, f"1h {choch} CHoCH (possible reversal)", now)
    return TriageResult(symbol, False, mtf.bias, labels, f"higher timeframes {mtf.bias}: no directional bias", now)


def prioritize(passed: list[str], priority_first: list[str], cap: int, cursor: int) -> tuple[list[str], int]:
    """Open positions / previously interesting coins first, then a ROTATING slice of the rest so every coin gets scanned."""
    first = list(dict.fromkeys(priority_first))
    rest = [s for s in passed if s not in first]
    room = max(0, cap - len(first))
    if not rest or room == 0:
        return first[:cap], cursor
    start = cursor % len(rest)
    rotated = rest[start:] + rest[:start]
    return first + rotated[:room], cursor + room
