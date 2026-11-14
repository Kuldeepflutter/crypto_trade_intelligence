"""Liquidation-cluster providers.

No provider data is ever invented. `NullProvider` returns nothing; the strategy then treats the
heatmap as unavailable. CoinGlass (or another licensed vendor) must be connected through
`HttpJsonProvider` with a field map that matches the vendor's CURRENT documented schema (verify it
yourself), or export their data to JSON and use `FileHeatmapProvider`.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from src.models import LiquidationCluster
from src.utils import is_finite


class LiquidationProvider(Protocol):
    name: str

    async def fetch_clusters(self, symbol: str) -> list[LiquidationCluster]: ...


def _ts(v) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_clusters(items: list[dict], fm: dict, source: str) -> list[LiquidationCluster]:
    out = []
    for it in items:
        try:
            price, notional = float(it[fm["price"]]), float(it[fm["notional"]])
        except (KeyError, TypeError, ValueError):
            continue
        if not (is_finite(price) and is_finite(notional)) or price <= 0 or notional <= 0:
            continue
        side = str(it.get(fm.get("side", "side"), "unknown")).lower()
        out.append(LiquidationCluster(price, notional, side if side in ("long", "short") else "unknown",
                                      _ts(it.get(fm.get("timestamp", "timestamp"))), source))
    return out


class NullProvider:
    name = "none"

    async def fetch_clusters(self, symbol: str) -> list[LiquidationCluster]:
        return []


class FileHeatmapProvider:
    """Reads JSON exported from a licensed provider: [{price, notional_usd, side, timestamp}, ...]."""
    name = "file"

    def __init__(self, path_template: str, field_map: dict):
        self.path_template, self.fm = path_template, field_map

    async def fetch_clusters(self, symbol: str) -> list[LiquidationCluster]:
        p = Path(self.path_template.format(symbol=symbol))
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get(self.fm.get("items", "data"), [])
        return parse_clusters(items, self.fm, f"file:{p.name}")


class HttpJsonProvider:
    """Generic licensed-provider adapter driven by a URL template and a field map."""
    name = "http_json"

    def __init__(self, http, url_template: str, api_key: str | None, api_key_header: str, field_map: dict, ttl: int = 120):
        self.http, self.url, self.key, self.hdr, self.fm, self.ttl = http, url_template, api_key, api_key_header, field_map, ttl

    async def fetch_clusters(self, symbol: str) -> list[LiquidationCluster]:
        if not self.key:
            raise RuntimeError("LIQ_API_KEY not set")
        data = await self.http.get_json(self.url.format(symbol=symbol), headers={self.hdr: self.key}, ttl=self.ttl, absolute=True)
        items = data if isinstance(data, list) else data.get(self.fm.get("items", "data"), [])
        return parse_clusters(items, self.fm, "http_json")
