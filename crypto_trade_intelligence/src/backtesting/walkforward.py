"""Walk-forward optimisation and a train / validation / out-of-sample split.

Parameters are chosen on the TRAIN window only and then evaluated on data the chooser never saw.
Only out-of-sample numbers should be read as evidence; in-sample numbers are reported for context.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from config import AppConfig
from src.backtesting.engine import Backtester
from src.backtesting.metrics import compute_metrics

DEFAULT_GRID = {"min_score": [60.0, 70.0], "stop_atr_buffer": [0.3, 0.5]}


def three_way_split(start: int, end: int, train: float = 0.5, val: float = 0.25) -> dict[str, tuple[int, int]]:
    n = end - start
    a, b = start + int(n * train), start + int(n * (train + val))
    return {"train": (start, a), "validation": (a, b), "out_of_sample": (b, end)}


def _with(cfg: AppConfig, params: dict) -> AppConfig:
    c = cfg.model_copy(deep=True)
    for k, v in params.items():
        setattr(c.strategy, k, v)
    return c


def _run(cfg, df5, params, window, kw):
    bt = Backtester(_with(cfg, params), df5, **kw)
    return bt.run(window[0], window[1])


def select_params(cfg, df5, grid, window, kw, min_trades: int = 8):
    best, best_m, rows = None, None, []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        m = _run(cfg, df5, params, window, kw).metrics
        rows.append({"params": params, **m})
        score = m["expectancy_r"] if (m["trades"] or 0) >= min_trades and m["expectancy_r"] is not None else float("-inf")
        if best is None or score > (best_m or float("-inf")):
            best, best_m = params, score
    return best, rows


@dataclass
class WalkForwardResult:
    folds: list[dict] = field(default_factory=list)
    oos_trades: list[dict] = field(default_factory=list)
    oos_metrics: dict = field(default_factory=dict)
    note: str = ""


def walk_forward(cfg: AppConfig, df5, grid=None, train_bars: int = 8640, test_bars: int = 2880, start: int | None = None,
                 min_trades: int = 8, **backtester_kw) -> WalkForwardResult:
    grid = grid or DEFAULT_GRID
    probe = Backtester(cfg, df5, **backtester_kw)
    s = max(start or 0, probe.warmup_bars)
    res = WalkForwardResult()
    while s + train_bars + test_bars <= len(df5) - 2:
        tr, te = (s, s + train_bars), (s + train_bars, s + train_bars + test_bars)
        params, rows = select_params(cfg, df5, grid, tr, backtester_kw, min_trades)
        if params is None:
            break
        oos = _run(cfg, df5, params, te, backtester_kw)
        res.folds.append({"train": tr, "test": te, "chosen": params, "oos": oos.metrics,
                          "in_sample_best": next((r for r in rows if r["params"] == params), None)})
        res.oos_trades += oos.trades
        s += test_bars
    res.oos_metrics = compute_metrics(res.oos_trades, cfg.risk.account_equity)
    res.note = "Only out-of-sample trades are aggregated here. Compare against the baselines before trusting any edge."
    return res
