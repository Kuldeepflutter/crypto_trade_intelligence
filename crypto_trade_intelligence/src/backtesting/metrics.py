"""Backtest statistics and simple baselines."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_metrics(trades: list[dict], start_equity: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": None, "expectancy_r": None, "avg_r": None, "profit_factor": None,
                "max_drawdown_pct": 0.0, "net_pnl": 0.0, "return_pct": 0.0, "total_fees": 0.0}
    pnl = np.array([t["pnl"] for t in trades], float)
    r = np.array([t["r_multiple"] for t in trades], float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = start_equity + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[start_equity, eq])[1:]
    dd = ((peak - eq) / peak).max() * 100
    return {"trades": n, "win_rate": float((pnl > 0).mean()), "expectancy_r": float(r.mean()), "avg_r": float(r.mean()),
            "expectancy_usd": float(pnl.mean()), "profit_factor": (float(wins / losses) if losses > 0 else None),
            "max_drawdown_pct": float(dd), "net_pnl": float(pnl.sum()), "return_pct": float(pnl.sum() / start_equity * 100),
            "total_fees": float(sum(t.get("fees", 0.0) for t in trades))}


def by_group(trades: list[dict], key: str, start_equity: float) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for t in trades:
        groups.setdefault(str(t.get(key)), []).append(t)
    return {k: compute_metrics(v, start_equity) for k, v in groups.items()}


def baseline_buy_hold(df5: pd.DataFrame, fee: float, slip_pct: float) -> dict:
    c = df5["close"]
    gross = c.iloc[-1] / c.iloc[0] - 1
    cost = 2 * (fee + slip_pct / 100)
    return {"name": "buy & hold (1x)", "return_pct": float((gross - cost) * 100),
            "max_drawdown_pct": float(((c.cummax() - c) / c.cummax()).max() * 100)}


def baseline_ema_cross(df5: pd.DataFrame, fee: float, slip_pct: float, fast: int = 20, slow: int = 50, allow_short: bool = True) -> dict:
    """1h EMA20/50 trend follower (long/short or long-only), 1x notional, same costs on every flip."""
    h = df5["close"].resample("1h").last().dropna()
    pos = np.sign(h.ewm(span=fast, adjust=False).mean() - h.ewm(span=slow, adjust=False).mean())
    if not allow_short:
        pos = pos.clip(lower=0)
    pos = pos.shift(1).fillna(0)                                  # act on the NEXT bar (no look-ahead)
    ret = h.pct_change().fillna(0) * pos - pos.diff().abs().fillna(0) * (fee + slip_pct / 100)
    eq = (1 + ret).cumprod()
    return {"name": f"EMA{fast}/{slow} 1h {'long/short' if allow_short else 'long-only'} (1x)",
            "return_pct": float((eq.iloc[-1] - 1) * 100), "max_drawdown_pct": float(((eq.cummax() - eq) / eq.cummax()).max() * 100)}
