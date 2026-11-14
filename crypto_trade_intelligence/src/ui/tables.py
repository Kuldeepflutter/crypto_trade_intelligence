"""Pure data-preparation for the dashboard (testable without Streamlit/Plotly)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from config import AppConfig
from src.models import MarketSnapshot, RatioKind
from src.utils import fmt_price


def _tp(plan, i, tick=None):
    return fmt_price(plan.tps[i], tick) if plan and i < len(plan.tps) else ""


def rows_to_frame(rows) -> pd.DataFrame:
    out = []
    for r in rows:
        ev, p = r.ev, r.ev.plan
        ca, cb = ev.cluster_above, ev.cluster_below
        out.append({
            "Symbol": r.symbol, "Price": ev.price, "24h Vol (USDT)": ev.volume_24h_usd,
            "5m": ev.trend_labels.get("5m", ""), "15m": ev.trend_labels.get("15m", ""),
            "1h": ev.trend_labels.get("1h", ""), "4h": ev.trend_labels.get("4h", ""),
            f"L/S ratio ({ev.ratio_kind})": None if not ev.ratio or ev.ratio.value is None else round(ev.ratio.value, 3),
            "OI chg %": None if ev.oi_change_pct is None else round(ev.oi_change_pct, 2),
            "Funding": ev.funding_rate,
            "Cluster above (est.)": "" if not ca else f"{ca['price']:.6g} (+{ca['distance_pct']:.2f}%)",
            "Cluster below (est.)": "" if not cb else f"{cb['price']:.6g} (-{cb['distance_pct']:.2f}%)",
            "Direction": ev.direction.value if ev.direction else "", "Outcome": ev.outcome.value,
            "Score": None if ev.score is None else round(ev.score, 1),
            "Entry": p.entry if p else None, "SL": p.stop_loss if p else None,
            "TP1": _tp(p, 0), "TP2": _tp(p, 1), "TP3": _tp(p, 2),
            "R/R (net)": None if not p else round(p.rr_net, 2),
            "Rejection reason": " | ".join((ev.rejections + ev.other_notes)[:3]),
            "Reduced features": ", ".join(ev.reduced_features),
            "Action": r.action,
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    order = {"TRADE SETUP": 0, "WATCH / WAIT FOR CONFIRMATION": 1, "NO TRADE": 2}
    return df.assign(_o=df["Outcome"].map(order).fillna(3), _s=-df["Score"].fillna(0)).sort_values(["_o", "_s", "Symbol"]).drop(columns=["_o", "_s"]).reset_index(drop=True)


def triage_frame(triage: dict) -> pd.DataFrame:
    rows = [{"Symbol": t.symbol, "Bias": t.bias, "1h": t.labels.get("1h", ""), "4h": t.labels.get("4h", ""),
             "Deep-scan eligible": t.passed, "Reason": t.reason, "Checked": t.at.isoformat()} for t in triage.values()]
    return pd.DataFrame(rows).sort_values(["Deep-scan eligible", "Symbol"], ascending=[False, True]).reset_index(drop=True) if rows else pd.DataFrame()


def metric_freshness(snap: MarketSnapshot, cfg: AppConfig, now: datetime | None = None) -> pd.DataFrame:
    """Source / timestamp / freshness of each metric, as required for display next to every value."""
    now = now or snap.now
    kind = RatioKind(cfg.ratio.kind)
    rows = []

    def add(name, value, source, ts, max_age):
        if value is None or ts is None:
            rows.append({"Metric": name, "Value": value, "Source": source, "Timestamp": None, "Age (s)": None, "Status": "MISSING"})
            return
        age = (now - ts).total_seconds()
        rows.append({"Metric": name, "Value": value, "Source": source, "Timestamp": ts.isoformat(), "Age (s)": round(age),
                     "Status": "fresh" if age <= max_age else "STALE"})

    src = cfg.exchange.name
    add("Price", snap.price, f"{src}:ticker", snap.timestamps.get("price"), 30)
    add("Mark price", snap.mark_price, f"{src}:premiumIndex", snap.funding.timestamp if snap.funding else None, 30)
    add("Funding rate", snap.funding.value if snap.funding else None, snap.funding.source if snap.funding else src, snap.funding.timestamp if snap.funding else None, 8 * 3600)
    h = snap.ratio_history.get(kind)
    add(f"L/S ratio ({kind.value})", None if h is None or h.empty else float(h["ratio"].iloc[-1]), f"{src}:{kind.value}",
        None if h is None or h.empty else h.index[-1].to_pydatetime(), cfg.ratio.max_age_seconds)
    oi = snap.oi_history
    add("Open interest", None if oi is None or oi.empty else float(oi["oi"].iloc[-1]), f"{src}:openInterestHist",
        None if oi is None or oi.empty else oi.index[-1].to_pydatetime(), 900)
    ob = snap.orderbook
    add("Order book", None if ob is None else len(ob.bids), ob.source if ob else src, ob.timestamp if ob else None, 30)
    cl = snap.liquidation_clusters
    ts = max((c.timestamp for c in cl if c.timestamp), default=None) if cl else None
    add("Liquidation clusters (ESTIMATED)", None if not cl else len(cl), snap.liquidation_source, ts, cfg.strategy.heatmap_max_age_seconds)
    for tf, df in snap.candles.items():
        add(f"Candles {tf}", len(df), f"{src}:klines", df["close_time"].iloc[-1].to_pydatetime(),
            {"5m": 900, "15m": 2700, "1h": 10800, "4h": 43200}[tf])
    for name, why in snap.errors.items():
        rows.append({"Metric": f"ERROR {name}", "Value": None, "Source": src, "Timestamp": None, "Age (s)": None, "Status": why})
    return pd.DataFrame(rows)
