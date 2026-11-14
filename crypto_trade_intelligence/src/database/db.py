"""SQLite storage: signals, trades (journal), alerts. Synchronous and thread-safe via a lock."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, symbol TEXT NOT NULL, direction TEXT, outcome TEXT NOT NULL,
  score REAL, setup_kind TEXT, entry REAL, stop_loss REAL, tp1 REAL, tp2 REAL, tp3 REAL, rr REAL,
  accepted INTEGER NOT NULL, reasons TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ix_signals_symbol_ts ON signals(symbol, ts);
CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id INTEGER, mode TEXT NOT NULL, symbol TEXT NOT NULL, direction TEXT NOT NULL,
  setup_kind TEXT, timeframe TEXT, opened_at TEXT NOT NULL, closed_at TEXT, entry_price REAL NOT NULL, avg_exit_price REAL,
  quantity REAL NOT NULL, stop_loss REAL, tp1 REAL, tp2 REAL, tp3 REAL, fees REAL DEFAULT 0, funding REAL DEFAULT 0,
  pnl REAL, r_multiple REAL, mae REAL, mfe REAL, status TEXT NOT NULL, exit_reason TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ix_trades_status ON trades(status, mode);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, symbol TEXT NOT NULL, direction TEXT NOT NULL,
  fingerprint TEXT NOT NULL, entry REAL, stop_loss REAL, score REAL, sent INTEGER NOT NULL, note TEXT);
CREATE INDEX IF NOT EXISTS ix_alerts_fp ON alerts(fingerprint, ts);
"""


def _iso(d: datetime | None) -> str | None:
    return None if d is None else d.astimezone(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._con.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._con.execute(sql, args)
            self._con.commit()
            return cur

    def query(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, args).fetchall()]

    def df(self, sql: str, args: tuple = ()) -> pd.DataFrame:
        return pd.DataFrame(self.query(sql, args))

    # ---- signals
    def record_signal(self, ev, accepted: bool) -> int:
        plan = ev.plan
        tps = (plan.tps + [None, None, None])[:3] if plan else [None] * 3
        reasons = ev.rejections if not accepted else ev.passed
        cur = self._exec(
            "INSERT INTO signals(ts,symbol,direction,outcome,score,setup_kind,entry,stop_loss,tp1,tp2,tp3,rr,accepted,reasons,payload)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_iso(ev.timestamp), ev.symbol, ev.direction.value if ev.direction else None, ev.outcome.value, ev.score,
             ev.setup_kind, plan.entry if plan else None, plan.stop_loss if plan else None, *tps,
             plan.rr_net if plan else None, int(accepted), json.dumps(reasons), json.dumps(ev.to_dict(), default=str)))
        return int(cur.lastrowid)

    # ---- trades
    def open_trade(self, **f) -> int:
        cols = ",".join(f.keys())
        cur = self._exec(f"INSERT INTO trades({cols}) VALUES({','.join('?' * len(f))})", tuple(f.values()))
        return int(cur.lastrowid)

    def update_trade(self, trade_id: int, **f) -> None:
        sets = ",".join(f"{k}=?" for k in f)
        self._exec(f"UPDATE trades SET {sets} WHERE id=?", (*f.values(), trade_id))

    def open_trades(self, mode: str) -> list[dict]:
        return self.query("SELECT * FROM trades WHERE status='open' AND mode=?", (mode,))

    def realized_since(self, mode: str, since: datetime) -> float:
        r = self.query("SELECT COALESCE(SUM(pnl),0) AS p FROM trades WHERE mode=? AND status='closed' AND closed_at>=?", (mode, _iso(since)))
        return float(r[0]["p"])

    def realized_total(self, mode: str) -> float:
        r = self.query("SELECT COALESCE(SUM(pnl),0) AS p FROM trades WHERE mode=? AND status='closed'", (mode,))
        return float(r[0]["p"])

    # ---- alerts
    def record_alert(self, ts: datetime, symbol: str, direction: str, fingerprint: str, entry, stop, score, sent: bool, note: str = "") -> None:
        self._exec("INSERT INTO alerts(ts,symbol,direction,fingerprint,entry,stop_loss,score,sent,note) VALUES(?,?,?,?,?,?,?,?,?)",
                   (_iso(ts), symbol, direction, fingerprint, entry, stop, score, int(sent), note))

    def last_alert(self, fingerprint: str) -> dict | None:
        r = self.query("SELECT * FROM alerts WHERE fingerprint=? AND sent=1 ORDER BY ts DESC LIMIT 1", (fingerprint,))
        return r[0] if r else None

    def performance(self, by: str = "symbol") -> pd.DataFrame:
        if by not in ("symbol", "setup_kind", "timeframe", "direction"):
            raise ValueError("invalid grouping")
        return self.df(f"SELECT {by} AS grp, COUNT(*) AS trades, SUM(pnl) AS pnl, AVG(r_multiple) AS avg_r, "
                       f"AVG(pnl>0) AS win_rate, SUM(fees) AS fees, SUM(funding) AS funding, AVG(mae) AS avg_mae, AVG(mfe) AS avg_mfe "
                       f"FROM trades WHERE status='closed' GROUP BY {by}")
