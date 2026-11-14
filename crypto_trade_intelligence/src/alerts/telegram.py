"""Alert formatting, de-duplication/cooldown, and Telegram delivery."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

from config import AlertConfig, AppConfig
from src.database.db import Database
from src.execution.guards import pre_trade_check
from src.utils import fmt_price, log_event

log = logging.getLogger("alerts")


class Notifier(Protocol):
    async def send(self, text: str) -> bool: ...


class ConsoleNotifier:
    async def send(self, text: str) -> bool:
        print("\n--- ALERT (console; Telegram not configured) ---\n" + text + "\n")
        return True


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, http=None):
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"   # never logged
        self._chat, self._http = chat_id, http

    async def send(self, text: str) -> bool:
        if self._http is None:
            import httpx
            self._http = httpx.AsyncClient(timeout=10)
        try:
            r = await self._http.post(self._url, json={"chat_id": self._chat, "text": text})
            ok = r.status_code == 200
            if not ok:
                log_event(log, logging.WARNING, "telegram_send_failed", status=r.status_code)
            return ok
        except Exception as e:  # network errors must never crash the scan loop
            log_event(log, logging.WARNING, "telegram_send_error", error=type(e).__name__)
            return False


def format_alert(ev, tick=None) -> str:
    p = ev.plan
    tp = lambda i: fmt_price(p.tps[i], tick) if i < len(p.tps) else "-"
    cl = ev.cluster_above if ev.direction.value == "LONG" else ev.cluster_below
    cl_txt = f"{fmt_price(cl['price'], tick)} ({cl['distance_pct']:.2f}% away, ESTIMATED)" if cl else "n/a"
    ratio = f"{ev.ratio.value:.3f} ({ev.ratio.kind.value})" if ev.ratio and ev.ratio.value is not None else "n/a"
    oi = f"{ev.oi_change_pct:+.2f}%" if ev.oi_change_pct is not None else "n/a"
    reduced = f"\nReduced-feature signal (missing: {', '.join(ev.reduced_features)})" if ev.reduced_features else ""
    return (
        f"Symbol: {ev.symbol}\nDirection: {ev.direction.value}\nCurrent price: {fmt_price(ev.price, tick)}\n"
        f"Entry zone: {fmt_price(p.entry_zone[0], tick)} - {fmt_price(p.entry_zone[1], tick)}"
        + (f" (alt after confirmation: {fmt_price(p.alt_entry, tick)})" if p.alt_entry else "") + "\n"
        f"Stop-loss: {fmt_price(p.stop_loss, tick)}\nTP1 / TP2 / TP3: {tp(0)} / {tp(1)} / {tp(2)}\n"
        f"Reward-to-risk (net of costs): {p.rr_net:.2f}\nLong/short ratio: {ratio}\nOI change: {oi}\n"
        f"Trend alignment: " + ", ".join(f"{k} {v}" for k, v in ev.trend_labels.items()) + "\n"
        f"Nearest liquidation cluster: {cl_txt}\nSetup score: {ev.score:.1f}/100 (not a win probability)\n"
        f"Data timestamp: {ev.timestamp.isoformat()}\nReasons supporting the trade:\n"
        + "\n".join(f" - {r}" for r in ev.passed) + "\nInvalidation conditions:\n"
        f" - close beyond {fmt_price(p.stop_loss, tick)} invalidates the setup\n"
        f" - ratio/trend/structure conditions above no longer hold" + reduced)


class AlertManager:
    def __init__(self, cfg: AppConfig, db: Database, notifier: Notifier):
        self.cfg, self.db, self.notifier = cfg, db, notifier

    @staticmethod
    def fingerprint(ev) -> str:
        return f"{ev.symbol}|{ev.direction.value}|{ev.setup_kind}"

    def _meaningful_change(self, ev, last: dict) -> bool:
        a: AlertConfig = self.cfg.alerts
        if last.get("score") is not None and ev.score is not None and abs(ev.score - last["score"]) >= a.min_score_change:
            return True
        atr = ev.atr15 or 0.0
        return bool(atr and last.get("entry") is not None and abs(ev.plan.entry - last["entry"]) >= a.entry_shift_atr * atr)

    async def maybe_alert(self, ev, now: datetime, tick=None) -> tuple[bool, str]:
        # Gate 2 of 3 for the SHORT filter: re-validated right before notifying.
        ok, reasons = pre_trade_check(ev, self.cfg.ratio, self.cfg.strategy, now)
        if not ok:
            log_event(log, logging.INFO, "alert_suppressed_guard", symbol=ev.symbol, reasons=reasons)
            return False, "guard: " + "; ".join(reasons)
        fp = self.fingerprint(ev)
        last = self.db.last_alert(fp)
        if last:
            age_min = (now - datetime.fromisoformat(last["ts"])).total_seconds() / 60
            if age_min < self.cfg.alerts.cooldown_minutes and not self._meaningful_change(ev, last):
                log_event(log, logging.INFO, "alert_suppressed_duplicate", symbol=ev.symbol, fingerprint=fp, age_min=round(age_min, 1))
                return False, "duplicate within cooldown"
        sent = await self.notifier.send(format_alert(ev, tick))
        self.db.record_alert(now, ev.symbol, ev.direction.value, fp, ev.plan.entry, ev.plan.stop_loss, ev.score, sent)
        return sent, "sent" if sent else "delivery failed"
