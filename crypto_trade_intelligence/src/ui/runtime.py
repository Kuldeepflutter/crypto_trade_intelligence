"""Background scanner so the GUI never blocks on network I/O."""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone


class BackgroundRunner:
    def __init__(self, app):
        self.app = app
        self.rows: dict[str, object] = {}
        self.last_scan: datetime | None = None
        self.last_error: str | None = None
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="scanner")
        self._started = False

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._forever())

    async def _forever(self):
        self.app.start_feed()
        while True:
            try:
                for row in await self.app.scan_all():
                    self.rows[row.symbol] = row
                self.last_scan, self.last_error = datetime.now(timezone.utc), self.app.last_error
            except Exception as e:                         # keep the loop alive; surface the error in the UI
                self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            await asyncio.sleep(self.app.cfg.scan_interval_seconds)

    def start(self) -> "BackgroundRunner":
        if not self._started:
            self._started = True
            self._thread.start()
        return self

    def wait_for_first_scan(self, timeout: float = 30.0) -> bool:
        t0 = time.time()
        while self.last_scan is None and time.time() - t0 < timeout:
            time.sleep(0.1)
        return self.last_scan is not None
