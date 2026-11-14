"""Chart data preparation (pure) and Plotly figure construction."""
from __future__ import annotations

import pandas as pd

from config import StrategyConfig
from src.indicators.core import ema, vwap
from src.market_structure.swings import find_levels
from src.models import MarketSnapshot


def chart_data(snap: MarketSnapshot, ev, cfg: StrategyConfig, tf: str = "15m", bars: int = 200) -> dict:
    df = snap.candles.get(tf)
    if df is None or df.empty:
        return {}
    ind = pd.DataFrame({"ema20": ema(df["close"], cfg.ema_fast), "ema50": ema(df["close"], cfg.ema_mid),
                        "ema200": ema(df["close"], cfg.ema_slow), "vwap": vwap(df)}, index=df.index)
    atr_v = ev.atr15 or float(df["close"].iloc[-1]) * 0.003
    levels = [(l.price, l.touches) for l in find_levels(df, 0.5 * atr_v, cfg.swing_left, cfg.swing_right) if l.touches >= 2]
    clusters = []
    if snap.liquidation_clusters:
        clusters = [(c.price, c.notional_usd, c.side) for c in snap.liquidation_clusters]
    plan = ev.plan
    lines = []
    if plan:
        lines = [("Entry", plan.entry), ("Stop-loss", plan.stop_loss)] + [(f"TP{i + 1}", p) for i, p in enumerate(plan.tps)]
    return {"candles": df.tail(bars), "ind": ind.tail(bars), "levels": levels, "clusters": clusters, "plan_lines": lines,
            "oi": snap.oi_history, "ratio": snap.ratio_history, "funding": snap.funding_history,
            "events": snap.liquidation_events}


def build_figure(data: dict, symbol: str, tf: str):
    import plotly.graph_objects as go                      # lazy import
    from plotly.subplots import make_subplots
    c, ind = data["candles"], data["ind"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.55, 0.15, 0.15, 0.15],
                        subplot_titles=(f"{symbol} {tf}", "Volume", "Open interest", "Long/short ratio"))
    fig.add_trace(go.Candlestick(x=c.index, open=c["open"], high=c["high"], low=c["low"], close=c["close"], name="Price"), row=1, col=1)
    for col, color in (("ema20", "#f59e0b"), ("ema50", "#3b82f6"), ("ema200", "#a855f7"), ("vwap", "#14b8a6")):
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=col.upper(), line=dict(width=1, color=color)), row=1, col=1)
    for price, touches in data["levels"]:
        fig.add_hline(y=price, line=dict(width=1, dash="dot", color="#9ca3af"), annotation_text=f"S/R x{touches}", row=1, col=1)
    mx = max((n for _, n, _ in data["clusters"]), default=0) or 1
    for price, notional, side in data["clusters"]:
        fig.add_hline(y=price, line=dict(width=1 + 5 * notional / mx, color="rgba(239,68,68,0.45)"),
                      annotation_text=f"ESTIMATED liq cluster ${notional / 1e6:.1f}M ({side})", row=1, col=1)
    for name, price in data["plan_lines"]:
        color = "#ef4444" if name == "Stop-loss" else ("#22c55e" if name.startswith("TP") else "#e5e7eb")
        fig.add_hline(y=price, line=dict(width=1.5, color=color), annotation_text=name, row=1, col=1)
    fig.add_trace(go.Bar(x=c.index, y=c["volume"], name="Volume"), row=2, col=1)
    oi = data["oi"]
    if oi is not None and len(oi):
        fig.add_trace(go.Scatter(x=oi.index, y=oi["oi"], name="OI"), row=3, col=1)
    for kind, h in (data["ratio"] or {}).items():
        if len(h):
            fig.add_trace(go.Scatter(x=h.index, y=h["ratio"], name=f"L/S ({kind.value})"), row=4, col=1)
    fig.add_hline(y=1.0, line=dict(dash="dash", color="#f97316"), annotation_text="1.0 (SHORT gate)", row=4, col=1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def funding_figure(data: dict, symbol: str):
    import plotly.graph_objects as go
    f = data["funding"]
    fig = go.Figure()
    if f is not None and len(f):
        fig.add_trace(go.Bar(x=f.index, y=f["rate"] * 100, name="Funding %"))
    fig.update_layout(height=220, title=f"{symbol} funding-rate history (%)", margin=dict(l=10, r=10, t=40, b=10))
    return fig
