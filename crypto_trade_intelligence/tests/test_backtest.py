"""Acceptance criterion 13 (fees + slippage) and look-ahead protection."""
from datetime import timedelta

import numpy as np
import pandas as pd

from src.backtesting.engine import Backtester
from src.backtesting.metrics import baseline_buy_hold, baseline_ema_cross, compute_metrics
from src.backtesting.walkforward import three_way_split, walk_forward
from src.indicators.core import resample_ohlcv
from tests.helpers import *


def _run(cfg, df, setup_idx, **kw):
    bt = Backtester(cfg, df, "TESTUSDT", **kw)
    return bt.run(start=setup_idx, end=len(df) - 1, step_bars=3)


def test_windows_contain_no_future_information():
    cfg = default_cfg()
    df = base_uptrend(12000, seed=3)
    bt = Backtester(cfg, df)
    for i in (10500, 11111, 11500):
        w = bt._windows(i)
        t = df["close_time"].iloc[i]
        for tf, frame in w.items():
            assert frame["close_time"].max() <= t, tf
        trunc = Backtester(cfg, df.iloc[: i + 1])._windows(i)               # same view if the future did not exist
        for tf in w:
            assert w[tf].equals(trunc[tf]) or np.allclose(w[tf][["open", "high", "low", "close"]].to_numpy(),
                                                          trunc[tf][["open", "high", "low", "close"]].to_numpy()), tf


def test_13_backtest_includes_fees_and_slippage_and_is_labelled_reduced_feature():
    cfg = default_cfg()
    base, info = build_long_scenario(cfg)
    idx = len(base) - 1
    up = extend(base, 150, +0.12)
    res = _run(cfg, up, idx)
    assert res.metrics["trades"] == 1, (res.metrics, res.rejection_counts)
    t = res.trades[0]
    assert t["fees"] > 0 and t["pnl"] > 0 and t["exit_reason"].startswith("tp")
    assert any("heatmap" in f for f in res.reduced_features) and any("order-book" in f for f in res.reduced_features)
    assert any("ratio" in f for f in res.reduced_features)
    # identical market, zero costs -> strictly higher PnL (costs are really applied)
    free = default_cfg()
    free.risk.taker_fee = free.risk.maker_fee = free.risk.max_slippage_pct = 0.0
    res0 = _run(free, up, idx)
    assert res0.trades[0]["fees"] == 0 and res0.trades[0]["pnl"] > t["pnl"] / t["qty"] * res0.trades[0]["qty"] - 1e-9
    assert res0.trades[0]["r_multiple"] > t["r_multiple"]
    # entry is the NEXT bar's open plus adverse slippage, not the signal bar's close
    nxt_open = float(up["open"].iloc[idx + 1])
    assert abs(t["entry"] - nxt_open * (1 + cfg.risk.max_slippage_pct / 100)) < 1e-6
    down = _run(cfg, extend(base, 150, -0.12), idx)
    assert down.trades[0]["exit_reason"] == "stop_loss" and down.trades[0]["r_multiple"] < -1.0


def test_backtest_short_requires_a_historical_ratio_series():
    cfg = default_cfg()
    base, _ = build_short_scenario(cfg)
    idx = len(base) - 1
    down = extend(base, 150, -0.12)
    blocked = _run(cfg, down, idx)
    assert blocked.metrics["trades"] == 0
    assert any("SHORT blocked" in k for k in blocked.rejection_counts), blocked.rejection_counts
    assert any("ALL shorts" in f for f in blocked.reduced_features)
    ratio_idx = pd.date_range(end=down["close_time"].iloc[-1], periods=400, freq="5min")
    cfg.strategy.min_rr = 1.5                                                  # fixture R/R sits near 2.0 once tick-rounded and filled
    good = _run(cfg, down, idx, ratio=pd.DataFrame({"ratio": 1.4}, index=ratio_idx))
    assert good.metrics["trades"] == 1 and good.trades[0]["direction"] == "SHORT" and good.trades[0]["pnl"] > 0
    low = _run(cfg, down, idx, ratio=pd.DataFrame({"ratio": 0.9}, index=ratio_idx))
    assert low.metrics["trades"] == 0                                           # ratio < 1.0 -> no SHORT in history either


def test_metrics_and_baselines():
    trades = [{"pnl": 100, "r_multiple": 2, "fees": 1}, {"pnl": -50, "r_multiple": -1, "fees": 1}, {"pnl": -50, "r_multiple": -1.1, "fees": 1}]
    m = compute_metrics(trades, 10000)
    assert m["trades"] == 3 and abs(m["win_rate"] - 1 / 3) < 1e-9 and abs(m["profit_factor"] - 1.0) < 1e-9
    assert abs(m["expectancy_r"] - (2 - 1 - 1.1) / 3) < 1e-9 and m["max_drawdown_pct"] > 0 and m["total_fees"] == 3
    assert compute_metrics([], 10000)["trades"] == 0
    df = base_uptrend(3000)
    assert baseline_buy_hold(df, 0.0005, 0.05)["return_pct"] > -100
    assert "return_pct" in baseline_ema_cross(df, 0.0005, 0.05)


def test_split_is_chronological_and_non_overlapping():
    s = three_way_split(1000, 11000)
    assert s["train"][0] == 1000 and s["train"][1] == s["validation"][0] and s["validation"][1] == s["out_of_sample"][0]
    assert s["out_of_sample"][1] == 11000 and s["train"][1] - s["train"][0] == 5000


def test_walk_forward_without_enough_data_returns_no_folds():
    cfg = default_cfg()
    res = walk_forward(cfg, base_uptrend(11000, seed=2))
    assert res.folds == [] and res.oos_metrics["trades"] == 0
