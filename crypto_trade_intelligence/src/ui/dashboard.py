"""Streamlit dashboard.  Run with:  streamlit run src/ui/dashboard.py   (or: python app.py dashboard)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from config import Secrets, load_config
from src.pipeline import TradeIntelligence
from src.ui.charts import build_figure, chart_data, funding_figure
from src.ui.runtime import BackgroundRunner
from src.ui.tables import metric_freshness, rows_to_frame, triage_frame
from src.utils import setup_logging

st.set_page_config(page_title="Crypto Trade Intelligence", layout="wide")


@st.cache_resource
def get_runner() -> BackgroundRunner:
    cfg = load_config()
    setup_logging(cfg.storage.log_dir, cfg.storage.log_level)
    return BackgroundRunner(TradeIntelligence.from_config(cfg, Secrets.from_env())).start()


runner = get_runner()
app, cfg = runner.app, runner.app.cfg

st.title("Crypto Futures Trade Intelligence")
mode = cfg.execution.mode
st.caption(f"Mode: **{mode.upper()}**  |  exchange: {cfg.exchange.name}  |  ratio used: {cfg.ratio.kind} ({cfg.ratio.period})  |  "
           f"liquidation mode: {cfg.strategy.liquidation_mode}  |  scores are heuristics, not win probabilities.")
if mode == "live":
    st.error("LIVE ORDER PLACEMENT IS ENABLED in the configuration.")
killed, why = app.risk.kill_switch_active(pd.Timestamp.now(tz="UTC").to_pydatetime())
if killed:
    st.error(why)
if runner.last_error:
    st.warning(f"Last scan error: {runner.last_error}")
col1, col2 = st.columns([1, 5])
if col1.button("Refresh view"):
    st.rerun()
col2.write(f"Last scan: {runner.last_scan.isoformat() if runner.last_scan else 'waiting for first scan...'}")

if cfg.universe.mode == "all":
    u = app.universe_stats
    st.caption(f"All-coins mode (public data, no API key): {u.total} USDT perpetuals discovered -> {u.candidates} pass volume/age filters -> "
               f"{u.triaged_pass} have a higher-timeframe bias -> {u.deep_scanned} deep-scanned this cycle (rotating, cap "
               f"{cfg.universe.max_deep_scans_per_cycle}). Skipped: {u.skipped}")

tab_scan, tab_coin, tab_journal, tab_tri = st.tabs(["Market scanner", "Coin detail", "Trade journal", "Triage (all coins)"])

with tab_scan:
    rows = list(runner.rows.values())
    if not rows:
        st.info("Scanning... (data is collected in the background; press Refresh view)")
    else:
        df = rows_to_frame(rows)
        st.dataframe(df, hide_index=True)
        st.caption("Rejection reasons include why a SHORT was blocked (e.g. long/short ratio < 1.0, missing or stale). "
                   "'Reduced features' lists inputs that were unavailable for that signal.")

with tab_coin:
    options = list(rows_to_frame(list(runner.rows.values()))["Symbol"]) if runner.rows else []
    sym = st.selectbox("Symbol (best setups first)", options) if options else None
    tf = st.radio("Timeframe", ["5m", "15m", "1h", "4h"], index=1, horizontal=True)
    row, snap = (runner.rows.get(sym), app.snapshots.get(sym)) if sym else (None, None)
    if row is None or snap is None:
        st.info("No data yet for this symbol.")
    else:
        data = chart_data(snap, row.ev, cfg.strategy, tf)
        if data:
            st.plotly_chart(build_figure(data, sym, tf))
            st.plotly_chart(funding_figure(data, sym))
        st.subheader(f"Signal: {row.ev.outcome.value}" + (f" - {row.ev.direction.value}" if row.ev.direction else ""))
        c1, c2 = st.columns(2)
        c1.markdown("**Passed / supporting**\n\n" + "\n".join(f"- {x}" for x in row.ev.passed) or "-")
        c2.markdown("**Rejections / waiting for**\n\n" + "\n".join(f"- {x}" for x in (row.ev.rejections + row.ev.other_notes)) or "-")
        if row.ev.components:
            st.dataframe(pd.DataFrame(row.ev.components).T)
        if row.sizing is not None:
            st.write(f"Planned size: {row.sizing.quantity} (risk {row.sizing.risk_actual:.2f} of budget {row.sizing.risk_budget:.2f} USDT)"
                     + (f" - {row.sizing.rejected}" if row.sizing.rejected else ""))
        st.subheader("Data sources, timestamps and freshness")
        st.dataframe(metric_freshness(snap, cfg), hide_index=True)
        if snap.liquidation_events:
            st.subheader("ACTUAL liquidation events (public feed) - separate from estimated clusters")
            st.dataframe(pd.DataFrame([vars(e) for e in snap.liquidation_events[-50:]]))

with tab_journal:
    sig = app.db.df("SELECT ts, symbol, direction, outcome, score, setup_kind, entry, stop_loss, tp1, tp2, tp3, rr, accepted, reasons FROM signals ORDER BY id DESC LIMIT 500")
    st.subheader("Generated signals (accepted and rejected)")
    st.dataframe(sig, hide_index=True)
    st.subheader("Trades (paper / live)")
    st.dataframe(app.db.df("SELECT * FROM trades ORDER BY id DESC LIMIT 500").drop(columns=["payload"], errors="ignore"),
                 hide_index=True)
    for by in ("symbol", "setup_kind", "timeframe"):
        st.subheader(f"Performance by {by}")
        st.dataframe(app.db.performance(by), hide_index=True)

with tab_tri:
    st.caption("Cheap 1h/4h look at every candidate coin. Only coins with a clear higher-timeframe bias (or a fresh 1h CHoCH) get a full scan.")
    st.dataframe(triage_frame(app.triage), hide_index=True)
