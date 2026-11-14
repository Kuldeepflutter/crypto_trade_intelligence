"""Small async-safe TTL cache so repeated scans do not hammer exchange APIs."""
from __future__ import annotations

import time
from typing import Any, Callable


class TTLCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic, max_items: int = 2048):
        self._d: dict[Any, tuple[float, Any]] = {}
        self._clock, self._max = clock, max_items

    def get(self, key):
        hit = self._d.get(key)
        if hit is None:
            return None
        exp, val = hit
        if exp < self._clock():
            self._d.pop(key, None)
            return None
        return val

    def set(self, key, value, ttl: float) -> None:
        if len(self._d) >= self._max:
            now = self._clock()
            self._d = {k: v for k, v in self._d.items() if v[0] >= now}
            if len(self._d) >= self._max:
                self._d.pop(next(iter(self._d)))
        self._d[key] = (self._clock() + ttl, value)
