# Crypto Futures Trade Intelligence

Python application that scans USDT-perpetual futures, scores LONG/SHORT setups, builds entry / stop / take-profit
plans net of costs, sizes positions from a risk budget, alerts, paper-trades, and backtests.

**It does not predict the market and cannot guarantee a profitable trade.** Every output is evidence-based context with a
heuristic 0-100 score (not a calibrated win probability), explicit invalidation levels, and a logged reason for every
acceptance or rejection. Default mode is **analysis-only**: no orders can be placed unless you deliberately enable them.

---------------------------------------------------------------------------------------------------------------------

## 1. Verification status (please read)

| Area | Status |
|---|---|
| Strategy engine, SHORT filter, planner, risk/sizing, paper broker, simulator, backtester, alerts, DB, dashboard data builders | Covered by **81 automated tests** (below) |
| How the tests were run by the author | In a sandbox **without network access** that had only numpy/pandas/PyYAML. `pytest` and `pydantic` were replaced by tiny local stand-ins. Please run `python -m pytest` on your machine with the real packages first. |
| Binance REST/WebSocket calls, universe discovery, request-weight numbers, the keyless liquidation model on real data, `ccxt`, Telegram delivery, Streamlit/Plotly UI | **Not executed against real services.** Parsers, rate limiting, retry logic and the model are unit-tested with fakes/synthetic data; the live HTTP/WS wiring and UI rendering are unverified. Expect to fix small things on first run. |
| Live order placement (`src/execution/live.py`) | Tested only against a fake exchange object. **Never run against a real exchange.** Use testnet and tiny size first, and review every order it creates. |
| Strategy edge | **Unvalidated.** Tests prove the rules are enforced, not that the rules make money. Test: synthetic markets built to exercise each gate. Use the backtester + paper trading on real data. |

## 2. Quick start (Windows 10/11, PowerShell)

```powershell
cd crypto_trade_intelligence
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env                # optional: only needed for Telegram / liquidation provider / live trading
copy config.example.yaml config.yaml
python -m pytest                      # all tests should pass
python app.py check-config            # shows what is and is not available
python app.py scan                    # one scan (analysis-only by default)
python app.py dashboard               # Streamlit dashboard at http://localhost:8501
```

Paper trading: set `execution.mode: paper` in `config.yaml`, then `python app.py run` (or use the dashboard).
Paper mode has no code path to an exchange: fills are simulated from closed candles.

**No keys needed.** With an empty `.env` the app scans **every USDT perpetual** on Binance using public endpoints only,
estimates liquidation clusters with a keyless public-data model (section 4), alerts to the console, paper-trades and backtests.
Keys are only for optional extras: Telegram, a licensed heatmap vendor, live orders.

If your network/region blocks the exchange API (HTTP 451/403 etc.) the app reports the error per metric and produces no
signals - it never switches exchanges or fabricates data. Use a connector for an exchange available to you.

## 3. Commands

| Command | Purpose |
|---|---|
| `python app.py check-config` | Effective configuration, which keys are present, warnings |
| `python app.py scan` | One scan of **all coins** (funnel below), prints the scanner table + funnel statistics |
| `python app.py scan --symbols BTCUSDT,ETHUSDT` | Scan only a fixed list |
| `python app.py universe` | Discovered coins with their 1h/4h triage result |
| `python app.py run` | Continuous scan loop (alerts, paper trades, journal) |
| `python app.py dashboard` | Streamlit UI: scanner, coin detail (chart, heatmap overlay, freshness table), journal |
| `python app.py backtest --symbol BTCUSDT --days 120` | Download history (cached in `data/`) and backtest |
| `... backtest --split` | Parameters chosen on TRAIN only, reported on train / validation / out-of-sample |
| `... backtest --walk-forward` | Rolling walk-forward, aggregates only out-of-sample trades |
| `... backtest --offline` | Use cached `data/<SYMBOL>_5m.csv` |

Needs about 35 days of 5m history just for indicator warm-up (4h EMA200), so use `--days` of 60 or more.

## 4. Data sources and keys required per feature

| Feature | Source | Key needed |
|---|---|---|
| Candles, last price, 24h volume, funding (+history), open interest (+history), global & top-trader long/short ratios, order-book depth, tick/step/min-notional rules | Binance USDⓈ-M public REST | **None** |
| Mark price, **actual** liquidation events (public `forceOrder` stream) | Binance public WebSocket (`LiveFeed`) | None |
| **Estimated** liquidation clusters, default | `liquidation.provider: public_model`: computed locally from public OI history + long/short ratio + a leverage-mix assumption | **None** (lower confidence, see below) |
| Estimated liquidation clusters, vendor-grade | Licensed provider (e.g. CoinGlass) via `provider: file` or `http_json` | Provider key in `LIQ_API_KEY` (paid plan) |
| Telegram alerts | Telegram Bot API | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| Live orders | Binance USDⓈ-M via ccxt | `BINANCE_API_KEY/SECRET` (trade permission, **no withdrawal**) + all gates in section 7 |
| Other exchanges | `exchange.name: ccxt` | Market data only; **no long/short ratio**, so SHORTs are impossible there |

Heatmap integration: no vendor schema is hard-coded because it could not be verified offline. Either export the vendor's
data to `data/heatmaps/<SYMBOL>.json` (`[{price, notional_usd, side, timestamp}]`) for the `file` provider, or point the
`http_json` provider at the vendor URL and map the field names (`liquidation.field_map`) from the vendor's current docs.
Clusters older than `heatmap_max_age_seconds` are discarded; clusters without a timestamp are discarded. Estimated
clusters and actual events are stored and displayed separately.

### 4b. All coins, public data only

`universe.mode: all` (default) discovers every TRADING USDT-margined perpetual from `exchangeInfo` and funnels them so 300+ coins
fit inside the public rate limits (a request-weight limiter and a separate `/futures/data` limiter make the client wait
rather than risk a 429/418 ban):

1. **Discover + filter (2 cheap calls):** bulk 24h tickers; drop coins below `strategy.min_volume_24h_usd` (default 20M USDT; set `0`
   for literally everything), listed less than `min_listing_days`, in `universe.exclude`, or beyond `max_symbols` (highest volume first).
2. **Triage (about 4 weight per coin, cached 15 min):** 1h + 4h candles only. Coins with a clear higher-timeframe bias or a fresh
   1h CHoCH continue; range-bound/conflicting coins stop here.
3. **Deep scan (about 35 weight per coin):** full 5m-4h analysis, ratio, OI, funding, order book, liquidation context, plan, score.
   At most `max_deep_scans_per_cycle` per cycle; open positions and recently interesting coins first, the rest **rotate** so every
   eligible coin is covered over time.

Coins that cannot be read (new listing, missing history, API error) are skipped with a recorded reason, never guessed.
Raise `max_symbols` / `max_deep_scans_per_cycle` if you want faster full coverage; the limiter will pace requests.
Notes: Binance may block some regions (HTTP 451/403); then no coin can be scanned and the app says so.

**The keyless liquidation model (`public_model`) is an estimate, not vendor data.** Binance does not publish liquidation levels, so
`src/liquidation/model.py` reconstructs a plausible distribution from public data: each rise in open interest is treated as new
exposure at that bar's typical price, split into longs/shorts by the public long/short **account** ratio, spread across leverage
tiers by `liquidation.model.leverage_mix` (an assumption you can edit), converted to liquidation prices, then exposure whose
liquidation price was already traded through is removed. Positions opened before the ~5-day window, cross-margin effects and
account-vs-notional mismatch are not modelled. Every cluster is flagged `estimated`, every signal that used it carries a
"lower confidence" label, and it is only one confirmation among several. It was validated for internal consistency on synthetic data
only; whether it improves results is unproven. If you later obtain a vendor key, switch `provider` to `http_json`/`file`.
The model needs the ratio and OI history; coins without them get no heatmap and (with `liquidation_mode: required`) no trade.
Actual forced liquidations from Binance's public `forceOrder` stream (a sample, not every event) are shown separately.

## 5. The mandatory SHORT filter

`src/positioning/ratio.py::short_entry_allowed` / `ratio_gate`, threshold `SHORT_MIN_RATIO = 1.0` (fixed in code, deliberately
not configurable). A SHORT is prohibited when the ratio is below 1.0, **or missing, stale, non-finite, has no timestamp, or is a
different ratio kind** than configured (global accounts / top-trader accounts / top-trader positions are never mixed).
A ratio >= 1.0 only removes this one restriction. It authorizes nothing by itself: trend, trigger, volume, structure,
liquidation context, OI, depth and net R/R must all still pass.

Enforced at three independent points, each re-validating freshness at the moment of use:
1. **Strategy engine** - before any SHORT candidate is generated (no plan or score is even computed).
2. **Alert manager** - before notifying.
3. **Execution** - before every paper fill and before every live order (live re-fetches the ratio).

Rejected SHORTs are recorded in the journal, the dashboard "Rejection reason" column and the structured log
(e.g. `SHORT: SHORT blocked: global_account long/short ratio 0.850 < 1.0`) and are never sent as alerts.

## 6. How a decision is made

1. **Validate data** (finite, fresh, enough history, price consistent with candles) - otherwise NO TRADE.
2. **Higher-timeframe bias** from 4h + 1h (EMA 20/50/200, swing structure, ADX, VWAP). Conflict -> WATCH, never forced.
   A reversal is accepted only as 1h CHoCH + 15m agreement + 4h not strongly opposed.
3. **Trigger from price action only**: liquidity-sweep + reclaim, breakout-retest, or support/resistance bounce.
   A liquidation cluster can never trigger a trade.
4. **Confirmations**: trigger volume, 15m/5m structure, liquidation context, OI/price contradiction, spread and depth.
5. **Plan**: stop = structural invalidation + ATR buffer (floor/ceiling in ATR), up to three targets at real levels or
   estimated clusters, rejected if an opposing level blocks the path or net R/R (after fees, slippage, funding) < `min_rr`
   or the stop would be too close to estimated liquidation at your leverage.
6. **Score** (0-100, configurable weights; missing components are excluded, not scored as zero). Below `min_score` -> WATCH.
7. Outcome: **TRADE SETUP**, **WATCH / WAIT FOR CONFIRMATION**, or **NO TRADE**, plus reasons for both directions.

Heuristics you may want to tune or remove (they are not validated): the ratio/funding/OI scoring curves, the "reclaimed
sweep inside intact 1h structure confirms structure" rule, and the 3-touch obstacle rule. Also note the SHORT-needs-ratio>=1.0
rule is a crowd-positioning assumption supplied by the specification; the backtester can quantify its effect only where
ratio history exists (exchanges keep about 30 days).

## 7. Risk, paper and live

* Size = risk budget / (stop distance + fees + slippage), then capped by margin allocation, remaining exposure, exchange
  step / min qty / min notional (Decimal arithmetic, rounded **down**; if the minimum size would exceed the budget the
  trade is rejected, never enlarged). Leverage only changes margin, never the risk budget.
* Kill switch: realized daily loss >= `max_daily_loss_pct` blocks new entries until the next UTC day. Also max positions and
  max total exposure.
* Paper broker persists positions/trades in SQLite (survives restarts) with fees, funding, MAE/MFE, R-multiple.
* **Live** requires ALL of: `execution.mode: live`, env `ALLOW_LIVE_ORDERS=yes-i-accept-the-risk`, API keys, passing the
  pre-trade guard, risk-manager approval, price within `max_slippage_pct` of plan. Defaults to the exchange testnet
  setting. If protective orders fail after entry, the position is flattened immediately. Orders are reduce-only.
* Use read-only keys for analysis. Secrets are read from environment variables only and redacted from logs; the httpx logger is
  silenced because Telegram URLs contain the bot token.

## 8. Backtesting

Closed-candle, no look-ahead (higher timeframes are resampled from 5m and only complete bars are visible). Decisions occur
at a bar close and fill at the **next bar's open** plus adverse slippage; fees, slippage and funding are charged; if stop and
target are both touched in one bar the **stop is assumed first**; gaps through the stop fill at the worse open; the R/R is
re-priced at the actual fill. Rejected and watch-listed signals are recorded with reasons. Reports: trades, win rate,
expectancy (R), profit factor, max drawdown, per-regime breakdown, and baselines (buy & hold, EMA20/50).

Historical liquidation heatmaps and order books are normally unavailable, so those features are switched off and every
report is labelled **REDUCED-FEATURE**. Without a historical ratio series the mandatory filter blocks all SHORTs in the test,
and ratio history older than ~30 days does not exist on the exchange. Do not tune parameters on the whole sample; use
`--split` or `--walk-forward` and read only the out-of-sample numbers.

## 9. Acceptance criteria -> tests

| # | Criterion | Test |
|---|---|---|
| 1-2 | SHORT at ratio 0.85 / 0.99 rejected | `test_short_filter.py::test_1_*`, `test_2_*` |
| 3 | Ratio exactly 1.00 passes only the gate | `test_3_ratio_exactly_1_passes_only_the_gate` |
| 4 | Missing / stale / invalid ratio rejected | `test_4_missing_stale_or_invalid_ratio_is_rejected` |
| 5 | LONG not auto-approved by a SHORT rejection | `test_5_long_is_not_auto_approved_*` |
| 6 | Cluster alone cannot trigger | `test_strategy_engine.py::test_6_*` |
| 7 | Conflicting timeframes -> WAIT | `test_7_conflicting_timeframes_produce_wait` |
| 8 | SL/TP from valid market data | `test_8_stop_and_targets_come_from_valid_market_data` |
| 9 | Invalid R/R rejected | `test_9_invalid_reward_risk_is_rejected` |
| 10 | Size respects budget + exchange limits | `test_risk.py::test_10_*` |
| 11 | API failures create no prices/signals | `test_data_failures.py::test_11_*` |
| 12 | Duplicate alerts suppressed | `test_alerts.py::test_12_*` |
| 13 | Backtests include fees and slippage | `test_backtest.py::test_13_*` |
| 14 | Paper trading places no real orders | `test_paper_and_live.py::test_14_*` |
| 15 | Every accept/reject logged with reasons | `test_strategy_engine.py::test_15_*` |
| 16 | Unit tests for filters and risk math | all of the above + `test_indicators.py`, `test_data_layer.py` |
| extra | All-coins funnel, rate-limit weights, keyless model | `test_universe_and_model.py` |

## 10. Layout

```
app.py  config.py  requirements.txt  .env.example  config.example.yaml  pytest.ini
src/exchanges/      http client (rate limit, cache, Retry-After, backoff), Binance + generic ccxt connectors
src/data_feeds/     collector (snapshot, errors never filled in), builder, order-book metrics, cache, WebSocket feed
src/liquidation/    cluster ranking / sweep detection, keyless public model, providers (none / file / http_json)
src/universe.py     all-coins discovery, filters, 1h/4h triage, rotating deep-scan selection
src/indicators/     EMA, ATR, ADX, VWAP, volume ratio, resampling (custom, no pandas-ta)
src/market_structure/  swings, HH/HL/LH/LL, BOS/CHoCH, S/R levels, multi-timeframe trend
src/positioning/    ratio gate + changes, OI/price context
src/strategies/     triggers, planner (net R/R), scoring, engine
src/risk/           sizing (Decimal), liquidation distance, limits + kill switch
src/execution/      simulator (shared with backtests), guards, paper broker, live broker
src/backtesting/    engine, metrics + baselines, walk-forward, history download
src/alerts/         formatting, dedupe/cooldown, Telegram
src/database/       SQLite: signals, trades, alerts
src/ui/             Streamlit dashboard, chart/table builders, background scanner
tests/              synthetic-market fixtures + the suites above (fixtures are never used by the app)
```

## 11. Known limitations

* Candle-level simulation cannot know intrabar order; the stop-first rule is pessimistic by design.
* Binance 5m ratio/OI timestamps are used as published; alignment with candle times is approximate at 5-minute resolution.
* Funding is only charged when history is supplied; expected funding paid is included in planned R/R.
* Estimated liquidation levels are model outputs from your provider, not facts; weigh them accordingly.
* Single-symbol backtests; no portfolio-level interaction.
