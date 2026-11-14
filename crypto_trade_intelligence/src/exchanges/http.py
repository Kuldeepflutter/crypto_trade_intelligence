"""Rate-limited async JSON client: token bucket, cache, Retry-After and exponential backoff.

Failures raise ExchangeError. This layer never returns made-up data.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from typing import Any, Callable

from src.data_feeds.cache import TTLCache
from src.utils import log_event

log = logging.getLogger("http")


class ExchangeError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


class RateLimiter:
    def __init__(self, rate_per_s: float, clock: Callable[[], float] = time.monotonic, sleep=asyncio.sleep):
        self.interval, self._clock, self._sleep = 1.0 / rate_per_s, clock, sleep
        self._next = 0.0
        self._lock: asyncio.Lock | None = None

    async def acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = self._clock()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            await self._sleep(wait)

    def penalize(self, seconds: float) -> None:
        self._next = max(self._next, self._clock() + seconds)     # honour Retry-After for all callers


class WeightLimiter:
    """Sliding-window budget (e.g. Binance request weight per minute). Waits instead of risking a 418/429 ban."""

    def __init__(self, max_weight: float, window_s: float, clock: Callable[[], float] = time.monotonic, sleep=asyncio.sleep):
        self.max, self.window, self._clock, self._sleep = max_weight, window_s, clock, sleep
        self._events: deque[tuple[float, float]] = deque()
        self._used = 0.0
        self._lock: asyncio.Lock | None = None

    async def acquire(self, weight: float = 1.0) -> None:
        weight = min(weight, self.max)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            while True:
                now = self._clock()
                while self._events and self._events[0][0] <= now - self.window:
                    self._used -= self._events.popleft()[1]
                if self._used + weight <= self.max:
                    self._events.append((now, weight))
                    self._used += weight
                    return
                await self._sleep(max(self._events[0][0] + self.window - now, 0.01))


class HttpClient:
    RETRY_STATUS = {418, 429, 500, 502, 503, 504}

    def __init__(self, base_url: str, rps: float = 8.0, max_retries: int = 5, timeout: float = 10.0, client=None,
                 cache: TTLCache | None = None, sleep=asyncio.sleep, clock: Callable[[], float] = time.monotonic,
                 weight_per_minute: float = 1800, data_per_5min: float = 900):
        self.base, self.max_retries, self.timeout = base_url.rstrip("/"), max_retries, timeout
        self._client, self.cache, self._sleep = client, cache or TTLCache(clock), sleep
        self.limiter = RateLimiter(rps, clock, sleep)
        self.budgets = {"fapi": WeightLimiter(weight_per_minute, 60, clock, sleep),
                        "data": WeightLimiter(data_per_5min, 300, clock, sleep)}
        self.weight_per_minute = weight_per_minute
        self.requests_made = 0

    async def _ensure(self):
        if self._client is None:
            import httpx                                  # lazy: only needed for real network use
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def get_json(self, path: str, params: dict | None = None, ttl: float = 0, headers: dict | None = None,
                       absolute: bool = False, weight: float = 1.0, bucket: str = "fapi") -> Any:
        url = path if absolute else self.base + path
        key = (url, tuple(sorted((params or {}).items())))
        if ttl > 0:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        client = await self._ensure()
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self.limiter.acquire()
            await self.budgets[bucket].acquire(weight)
            try:
                self.requests_made += 1
                r = await client.get(url, params=params, headers=headers, timeout=self.timeout)
            except Exception as e:                                         # network / timeout
                last = e
                await self._backoff(attempt, None, type(e).__name__)
                continue
            if r.status_code == 200:
                used = r.headers.get("X-MBX-USED-WEIGHT-1M") if hasattr(r.headers, "get") else None
                if used and str(used).isdigit() and int(used) >= 0.9 * 2400:     # exchange says we are near its hard limit
                    self.limiter.penalize(15.0)
                    log_event(log, logging.WARNING, "weight_near_limit", used=int(used))
                try:
                    data = r.json()
                except ValueError as e:
                    raise ExchangeError(f"invalid JSON from {path}") from e
                if ttl > 0:
                    self.cache.set(key, data, ttl)
                return data
            if r.status_code in self.RETRY_STATUS:
                ra = r.headers.get("Retry-After") if hasattr(r.headers, "get") else None
                last = ExchangeError(f"HTTP {r.status_code} from {path}", r.status_code)
                await self._backoff(attempt, float(ra) if ra and str(ra).replace(".", "", 1).isdigit() else None, str(r.status_code))
                continue
            raise ExchangeError(f"HTTP {r.status_code} from {path}: {str(r.text)[:160]}", r.status_code)
        raise ExchangeError(f"request to {path} failed after {self.max_retries + 1} attempts: {last}") from last

    async def _backoff(self, attempt: int, retry_after: float | None, why: str) -> None:
        if attempt >= self.max_retries:
            return
        wait = retry_after if retry_after is not None else min(30.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.25)
        self.limiter.penalize(wait)
        log_event(log, logging.WARNING, "http_retry", reason=why, wait_s=round(wait, 2), attempt=attempt + 1)
        await self._sleep(wait)
