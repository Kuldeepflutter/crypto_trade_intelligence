"""Long/short ratio handling, including the MANDATORY SHORT-entry gate.

The gate lives here (not in the UI) and is enforced three times: in the strategy engine before a
SHORT is generated, in the alert manager before a notification, and in the execution layer before
an order. All three call `ratio_gate`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from config import SHORT_MIN_RATIO
from src.models import RatioKind, RatioReading
from src.utils import is_finite


def short_entry_allowed(long_short_ratio: float | None) -> bool:
    """Spec function: SHORT is prohibited when the ratio is missing, invalid or below 1.0.

    Passing this function only removes ONE restriction; it never authorizes a SHORT by itself.
    """
    if long_short_ratio is None:
        return False
    if isinstance(long_short_ratio, bool) or not isinstance(long_short_ratio, (int, float)):
        return False
    if not math.isfinite(long_short_ratio) or long_short_ratio <= 0:
        return False
    if long_short_ratio < SHORT_MIN_RATIO:
        return False
    return True


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str


def ratio_gate(reading: RatioReading | None, expected_kind: RatioKind, now: datetime, max_age_seconds: float) -> GateResult:
    """Evaluate the SHORT gate on a specific ratio kind with freshness checking."""
    if reading is None or reading.value is None:
        return GateResult(False, f"SHORT blocked: {expected_kind.value} long/short ratio unavailable")
    if reading.kind != expected_kind:
        return GateResult(False, f"SHORT blocked: ratio kind mismatch (got {reading.kind.value}, expected {expected_kind.value}); metrics are never mixed")
    if not is_finite(reading.value) or reading.value <= 0:
        return GateResult(False, f"SHORT blocked: invalid ratio value {reading.value!r}")
    if reading.timestamp is None:
        return GateResult(False, "SHORT blocked: ratio has no timestamp (cannot verify freshness)")
    age = (now - reading.timestamp).total_seconds()
    if age > max_age_seconds or age < -120:
        return GateResult(False, f"SHORT blocked: ratio stale/invalid timestamp (age {age:.0f}s > {max_age_seconds:.0f}s)")
    if not short_entry_allowed(reading.value):
        return GateResult(False, f"SHORT blocked: {expected_kind.value} long/short ratio {reading.value:.3f} < {SHORT_MIN_RATIO}")
    return GateResult(True, f"ratio gate passed ({expected_kind.value} {reading.value:.3f} >= {SHORT_MIN_RATIO}); other confirmations still required")


def latest_reading(df: pd.DataFrame | None, kind: RatioKind, source: str, period: str = "") -> RatioReading | None:
    if df is None or df.empty:
        return None
    row = df.iloc[-1]
    return RatioReading(float(row["ratio"]), kind, df.index[-1].to_pydatetime(), source, period)


def ratio_changes(df: pd.DataFrame | None, now: datetime, minutes=(5, 15, 60)) -> dict[str, float | None]:
    """Absolute change in the ratio over N minutes. None where the history does not reach back far enough."""
    out: dict[str, float | None] = {f"{m}m": None for m in minutes}
    if df is None or len(df) < 2:
        return out
    last_t, last_v = df.index[-1], float(df["ratio"].iloc[-1])
    for m in minutes:
        target = last_t - pd.Timedelta(minutes=m)
        past = df.loc[:target]
        if past.empty or (target - past.index[-1]) > pd.Timedelta(minutes=max(m, 10)):
            continue
        out[f"{m}m"] = last_v - float(past["ratio"].iloc[-1])
    return out


def ratio_context_score(direction_sign: int, ratio: float | None, crowded: float) -> float | None:
    """Heuristic 0..1 contextual score (NOT a hard filter for LONGs).

    SHORT: more crowded longs (higher ratio) is more supportive. LONG: crowded shorts (low ratio)
    is more supportive, very crowded longs is a caution.
    """
    import numpy as np
    if not is_finite(ratio):
        return None
    if direction_sign < 0:
        return float(np.interp(ratio, [SHORT_MIN_RATIO, 1.3, crowded], [0.4, 0.8, 1.0]))
    return float(np.interp(ratio, [0.8, 1.0, crowded], [1.0, 0.6, 0.1]))
