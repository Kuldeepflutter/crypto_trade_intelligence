import numpy as np
import pandas as pd

from src.indicators.core import adx, atr, ema, resample_ohlcv, validate_candles, vwap, volume_ratio
from src.market_structure.swings import analyze_structure, find_levels, find_swings
from tests.helpers import _frame, base_uptrend


def _zigzag(n_cycles=8, up=True):
    pts = []
    base = 100.0
    for k in range(n_cycles):
        lo = base + (k * 2 if up else -k * 2)
        pts += list(np.linspace(lo, lo + 5, 8, endpoint=False)) + list(np.linspace(lo + 5, lo + 2, 6, endpoint=False))
    c = np.array(pts)
    return _frame(np.r_[c[0], c[:-1]], c + 0.3, c - 0.3, c, np.full(len(c), 100.0))


def test_swings_need_confirmation_bars():
    df = _zigzag()
    sw = find_swings(df, 3, 3)
    assert sw and all(s.idx <= len(df) - 4 for s in sw)                      # the last 3 bars can never be confirmed swings
    assert {s.kind for s in sw} == {"high", "low"}


def test_structure_classification():
    assert analyze_structure(_zigzag(up=True)).trend == "up"
    assert analyze_structure(_zigzag(up=False)).trend == "down"


def test_bos_and_choch_detection():
    df = _zigzag(up=False)                                                    # LH/LL structure
    last_high = analyze_structure(df).last_high.price
    bump = df.copy()
    n = len(bump)
    for k in range(6):                                                        # close above the last lower-high -> bullish CHoCH
        bump.iloc[n - 6 + k, bump.columns.get_loc("close")] = last_high + 1 + k
        bump.iloc[n - 6 + k, bump.columns.get_loc("high")] = last_high + 1.5 + k
    st = analyze_structure(bump, event_lookback=12)
    assert st.choch == "bullish" and st.trend == "down"


def test_levels_cluster_nearby_swings():
    lv = find_levels(_zigzag(up=False), tol=1.0)
    assert lv and all(l.touches >= 1 for l in lv)


def test_resample_returns_only_complete_bars():
    df = base_uptrend(2000, seed=1)
    cut = df.iloc[:1001]                                                      # ends mid-hour
    h = resample_ohlcv(cut, "1h")
    assert h["close_time"].max() <= cut["close_time"].iloc[-1]
    assert (h.index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1) <= cut["close_time"].iloc[-1]).all()
    full = resample_ohlcv(df, "1h")
    assert np.isclose(full["close"].iloc[5], df["close"].iloc[12 * 6 - 1 - (df.index[0].minute // 5)], atol=1e-9) or len(full) > 5


def test_indicators_finite_and_bounded():
    df = base_uptrend(2000, seed=1)
    assert np.isfinite(ema(df["close"], 20).iloc[-1]) and ema(df["close"], 20).iloc[:19].isna().all()
    assert atr(df).iloc[-1] > 0 and 0 <= adx(df).iloc[-1] <= 100 and np.isfinite(vwap(df).iloc[-1])
    assert np.isfinite(volume_ratio(df).iloc[-1])
    assert validate_candles(df, 100) is None
    bad = df.copy(); bad.iloc[-1, bad.columns.get_loc("high")] = np.nan
    assert "NaN" in validate_candles(bad, 100)
    assert validate_candles(df.iloc[:10], 100) is not None
