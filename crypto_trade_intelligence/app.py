    #!/usr/bin/env python3
"""Crypto Trade Intelligence - command line entry point.

  python app.py check-config            show effective configuration and which features are available
  python app.py scan                    one scan of ALL USDT perpetuals (public data, no API key)
  python app.py universe                list discovered coins and their 1h/4h triage
  python app.py run                     continuous scanning loop
  python app.py dashboard               launch the Streamlit dashboard
  python app.py backtest --symbol BTCUSDT --days 120 [--split | --walk-forward]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from config import ROOT, Secrets, load_config
from src.utils import setup_logging


def _print_rows(rows) -> None:
    import pandas as pd
    from src.ui.tables import rows_to_frame
    if not rows:
        print("No results.")
        return
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)
    print(rows_to_frame(rows).to_string(index=False))


def cmd_check(args) -> int:
    cfg, sec = load_config(args.config), Secrets.from_env()
    print(f"mode: {cfg.execution.mode} (default is analysis-only; orders need mode=live + ALLOW_LIVE_ORDERS + API keys)")
    print(f"exchange: {cfg.exchange.name} (public endpoints: NO API KEY NEEDED for scanning, alerts-to-console, paper trading, backtests)")
    if cfg.universe.mode == "all":
        print(f"universe: ALL USDT perpetuals (volume >= {cfg.strategy.min_volume_24h_usd:,.0f} USDT/24h, listed >= {cfg.universe.min_listing_days}d, "
              f"top {cfg.universe.max_symbols} by volume; {cfg.universe.max_deep_scans_per_cycle} deep scans per cycle)")
    else:
        print(f"universe: fixed list {', '.join(cfg.symbols)}")
    print(f"ratio kind used for the SHORT filter: {cfg.ratio.kind} ({cfg.ratio.period}, max age {cfg.ratio.max_age_seconds}s)")
    print(f"liquidation provider: {cfg.liquidation.provider}   liquidation mode: {cfg.strategy.liquidation_mode}")
    if cfg.liquidation.provider == "public_model":
        print("  -> keyless MODEL from public OI + long/short ratio + leverage-mix assumptions (estimate, lower confidence; labelled on every signal)")
    print(f"telegram: {'enabled' if cfg.alerts.telegram_enabled else 'disabled'}  (token set: {bool(sec.telegram_bot_token)})")
    print(f"API keys present: {bool(sec.binance_api_key and sec.binance_api_secret)}  liq API key: {bool(sec.liq_api_key)}")
    if cfg.strategy.liquidation_mode == "required" and cfg.liquidation.provider == "none":
        print("WARNING: liquidation_mode=required but no liquidation provider is configured -> no TRADE SETUP can ever qualify. "
              "Configure a provider or set strategy.liquidation_mode: optional (signals are then labelled reduced-feature).")
    return 0


def _make_app(args):
    from src.pipeline import TradeIntelligence
    cfg = load_config(args.config)
    if getattr(args, "symbols", None):
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        cfg.universe.mode = "fixed"
    setup_logging(str(ROOT / cfg.storage.log_dir), cfg.storage.log_level)
    return TradeIntelligence.from_config(cfg, Secrets.from_env())


def _print_stats(ti) -> None:
    if ti.cfg.universe.mode == "all":
        u = ti.universe_stats
        print(f"\nuniverse: {u.total} perpetuals -> {u.candidates} candidates -> {u.triaged_pass} with HTF bias -> {u.deep_scanned} deep-scanned; skipped: {u.skipped}")
    if ti.last_error:
        print(f"ERROR (no signals produced from missing data): {ti.last_error}")


def cmd_scan(args) -> int:
    ti = _make_app(args)
    _print_rows(asyncio.run(ti.scan_all()))
    _print_stats(ti)
    return 0


def cmd_universe(args) -> int:
    """List discovered coins and their 1h/4h triage (no deep scan, no orders)."""
    import pandas as pd
    from src.ui.tables import triage_frame
    ti = _make_app(args)
    ti.cfg.universe.max_deep_scans_per_cycle = 1
    asyncio.run(ti.scan_all())
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print(triage_frame(ti.triage).to_string(index=False))
    _print_stats(ti)
    return 0


def cmd_run(args) -> int:
    ti = _make_app(args)
    try:
        asyncio.run(ti.run_forever(on_rows=lambda rows: (_print_rows(rows), _print_stats(ti))))
    except KeyboardInterrupt:
        print("stopped")
    return 0


def cmd_dashboard(args) -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "ui" / "dashboard.py")])


def _report(res, title: str) -> None:
    m = res.metrics
    print(f"\n=== {title} ({res.start} -> {res.end}, {res.steps} decision points) ===")
    print("REDUCED-FEATURE BACKTEST - not used: " + "; ".join(res.reduced_features))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    for b in res.baselines:
        print(f"baseline {b['name']}: return {b['return_pct']:.2f}%  max DD {b['max_drawdown_pct']:.2f}%")
    print("by 4h regime:", {k: (v["trades"], v["expectancy_r"]) for k, v in res.by_regime.items()})
    print("top rejection reasons:", res.rejection_counts)
    print("Note: strategy return uses risk-based sizing; baselines use 1x notional - compare R-expectancy and drawdown, not raw returns.")


async def _backtest(args) -> int:
    from src.backtesting.data import cache_path, fetch_history_backwards, fetch_klines_paged, load_candles, save_candles
    from src.backtesting.engine import Backtester
    from src.backtesting.walkforward import DEFAULT_GRID, select_params, three_way_split, walk_forward
    from src.exchanges.binance import BinanceUSDM
    from src.exchanges.http import HttpClient
    from src.models import RatioKind
    cfg = load_config(args.config)
    setup_logging(str(ROOT / cfg.storage.log_dir), "WARNING")
    sym, kind = args.symbol.upper(), RatioKind(cfg.ratio.kind)
    http = HttpClient(cfg.exchange.base_url, cfg.exchange.requests_per_second, cfg.exchange.max_retries, cfg.exchange.timeout_s,
                      weight_per_minute=cfg.exchange.weight_per_minute, data_per_5min=cfg.exchange.data_requests_per_5min)
    conn = BinanceUSDM(http)
    path = cache_path(sym)
    if args.offline and path.exists():
        df5 = load_candles(path)
    else:
        print(f"downloading {args.days} days of 5m candles for {sym} ...")
        df5 = await fetch_klines_paged(conn, sym, "5m", args.days)
        save_candles(df5, path)
    kw = {"symbol": sym, "ratio_kind": kind}
    if not args.offline:
        days = min(args.days, 30)                                      # exchanges only keep ~30 days of ratio / OI history
        try:
            kw["ratio"] = await fetch_history_backwards(lambda n, e: conn.ratio_history(kind, sym, "5m", n, e), days)
            kw["oi"] = await fetch_history_backwards(lambda n, e: conn.open_interest_history(sym, "5m", n, e), days)
            kw["funding"] = await conn.funding_history(sym, 1000, int((df5.index[0]).timestamp() * 1000))
            kw["rules"] = await conn.symbol_rules(sym)
        except Exception as e:
            print(f"warning: positioning/rules history unavailable ({type(e).__name__}: {e}); continuing reduced-feature")
        for k in ("ratio", "oi"):
            if k in kw and (kw[k] is None or len(kw[k]) == 0):
                kw.pop(k)
        print("NOTE: ratio/OI history covers at most the last ~30 days; older bars have no ratio, so SHORTs are blocked there.")
    probe = Backtester(cfg, df5, **kw)
    if len(df5) < probe.warmup_bars + 3000:
        print(f"need at least {probe.warmup_bars + 3000} 5m bars ({(probe.warmup_bars + 3000) / 288:.0f} days) incl. indicator warm-up; have {len(df5)}")
        return 2
    if args.walk_forward:
        wf = walk_forward(cfg, df5, DEFAULT_GRID, args.train_bars, args.test_bars, **kw)
        print(json.dumps({"folds": wf.folds, "oos_metrics": wf.oos_metrics, "note": wf.note}, indent=2, default=str))
        return 0
    if args.split:
        sp = three_way_split(probe.warmup_bars, len(df5) - 2)
        params, rows = select_params(cfg, df5, DEFAULT_GRID, sp["train"], kw)
        print("parameters chosen on TRAIN only:", params)
        for name in ("train", "validation", "out_of_sample"):
            c = cfg.model_copy(deep=True)
            for k, v in params.items():
                setattr(c.strategy, k, v)
            r = Backtester(c, df5, **kw).run(*sp[name], step_bars=args.step)
            _report(r, name.upper() + (" (final hold-out: look at this once)" if name == "out_of_sample" else ""))
        return 0
    res = probe.run(step_bars=args.step)
    _report(res, f"{sym} full period")
    out = ROOT / "data" / f"backtest_{sym}.json"
    out.write_text(json.dumps({"metrics": res.metrics, "reduced_features": res.reduced_features, "trades": res.trades,
                               "rejection_counts": res.rejection_counts, "baselines": res.baselines}, indent=2, default=str))
    print(f"details written to {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml if present)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-config").set_defaults(fn=cmd_check)
    for name, fn in (("scan", cmd_scan), ("run", cmd_run), ("universe", cmd_universe)):
        p = sub.add_parser(name)
        p.add_argument("--symbols", help="comma separated list; switches to fixed-list mode (default: scan ALL coins)")
        p.set_defaults(fn=fn)
    sub.add_parser("dashboard").set_defaults(fn=cmd_dashboard)
    b = sub.add_parser("backtest")
    b.add_argument("--symbol", default="BTCUSDT")
    b.add_argument("--days", type=int, default=120)
    b.add_argument("--step", type=int, default=3, help="evaluate every N 5m bars")
    b.add_argument("--offline", action="store_true", help="use cached data/<SYMBOL>_5m.csv, no network")
    b.add_argument("--split", action="store_true", help="train / validation / out-of-sample report")
    b.add_argument("--walk-forward", action="store_true")
    b.add_argument("--train-bars", type=int, default=8640)
    b.add_argument("--test-bars", type=int, default=2880)
    b.set_defaults(fn=lambda a: asyncio.run(_backtest(a)))
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
