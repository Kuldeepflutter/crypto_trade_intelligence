"""Small shared helpers: finite checks, Decimal rounding, time, structured logging."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from pathlib import Path

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
TIMEFRAMES = ("5m", "15m", "1h", "4h")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def is_finite(x) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def to_decimal(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def round_to_step(value, step, mode: str = "down") -> Decimal:
    """Round to a multiple of `step` using Decimal (exchange precision safe)."""
    v, s = to_decimal(value), to_decimal(step)
    if s <= 0:
        raise ValueError("step must be > 0")
    rounding = {"down": ROUND_DOWN, "up": ROUND_UP, "nearest": ROUND_HALF_UP}[mode]
    return (v / s).to_integral_value(rounding=rounding) * s


def fmt_price(x, tick=None) -> str:
    if x is None or not is_finite(x):
        return "n/a"
    if tick is not None:
        d = to_decimal(tick).normalize()
        places = max(0, -d.as_tuple().exponent)
        return f"{float(x):.{places}f}"
    return f"{float(x):.6g}"


# ----------------------------------------------------------------- logging
_REDACT = ("secret", "token", "api_key", "apikey", "password", "signature")


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("***" if any(t in str(k).lower() for t in _REDACT) else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname, "logger": record.name, "event": record.getMessage()}
        fields = getattr(record, "fields", None)
        if fields:
            base.update(_redact(fields))
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


def log_event(logger: logging.Logger, level: int, event: str, **fields) -> None:
    logger.log(level, event, extra={"fields": fields})


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / "trade_intel.jsonl", encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    sh = logging.StreamHandler()
    sh.setFormatter(JsonFormatter())
    root.addHandler(fh)
    root.addHandler(sh)
    # httpx logs full request URLs at INFO (Telegram URLs contain the bot token) -> silence it.
    for noisy in ("httpx", "httpcore", "websockets", "ccxt"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
