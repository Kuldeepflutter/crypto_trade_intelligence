"""LIVE order placement via ccxt. DISABLED unless every gate below is satisfied.

Gates: execution.mode == "live"  AND  env ALLOW_LIVE_ORDERS == "yes-i-accept-the-risk"  AND  API
keys present  AND  the pre-trade guard (incl. the mandatory SHORT filter, re-checked on a FRESH ratio)
passes  AND  the risk manager approves.  Defaults to the exchange TESTNET.

NOTE: this module has not been exercised against a real exchange by the author. Run it on testnet
with tiny sizes first and review every order it creates.
"""
from __future__ import annotations

import logging
from datetime import datetime

from config import AppConfig, Secrets
from src.execution.guards import pre_trade_check
from src.models import RatioReading
from src.risk.manager import RiskManager
from src.risk.sizing import SizingResult
from src.utils import log_event

log = logging.getLogger("live")


class LiveTradingDisabled(RuntimeError):
    pass


def ccxt_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT:USDT" if symbol.endswith("USDT") else symbol


class LiveBroker:
    mode = "live"

    def __init__(self, cfg: AppConfig, secrets: Secrets, risk: RiskManager, exchange=None):
        if cfg.execution.mode != "live":
            raise LiveTradingDisabled("execution.mode is not 'live'")
        if not secrets.allow_live_orders:
            raise LiveTradingDisabled("env ALLOW_LIVE_ORDERS must be 'yes-i-accept-the-risk'")
        if not (secrets.binance_api_key and secrets.binance_api_secret) and exchange is None:
            raise LiveTradingDisabled("API key/secret not configured")
        self.cfg, self.risk = cfg, risk
        if exchange is None:
            import ccxt.async_support as ccxt   # lazy import
            exchange = ccxt.binanceusdm({"apiKey": secrets.binance_api_key, "secret": secrets.binance_api_secret,
                                         "enableRateLimit": True})
            if cfg.execution.exchange_testnet:
                exchange.set_sandbox_mode(True)
        self.ex = exchange

    async def place_bracket(self, ev, sizing: SizingResult, now: datetime, open_notionals: list[float],
                            fresh_ratio: RatioReading | None) -> dict:
        ok, reasons = pre_trade_check(ev, self.cfg.ratio, self.cfg.strategy, now, fresh_ratio)
        if not ok:
            raise LiveTradingDisabled("pre-trade guard: " + "; ".join(reasons))
        if not sizing.ok:
            raise LiveTradingDisabled("sizing: " + str(sizing.rejected))
        dec = self.risk.approve_entry(open_notionals, float(sizing.notional), now)
        if not dec.allowed:
            raise LiveTradingDisabled("risk manager: " + "; ".join(dec.reasons))
        plan, d, sym = ev.plan, ev.direction, ccxt_symbol(ev.symbol)
        side, opp = ("buy", "sell") if d.sign > 0 else ("sell", "buy")
        ticker = await self.ex.fetch_ticker(sym)
        px = float(ticker["last"])
        if d.sign * (px - plan.entry) / plan.entry * 100 > self.cfg.risk.max_slippage_pct:
            raise LiveTradingDisabled(f"price {px} moved beyond max slippage from planned entry {plan.entry}")
        qty = float(self.ex.amount_to_precision(sym, float(sizing.quantity)))
        try:
            await self.ex.set_margin_mode("isolated", sym)
        except Exception as e:                      # already isolated is reported as an error by some exchanges
            log_event(log, logging.INFO, "set_margin_mode_skipped", reason=str(e)[:120])
        await self.ex.set_leverage(int(self.cfg.risk.leverage), sym)
        entry = await self.ex.create_order(sym, "market", side, qty)
        orders = {"entry": entry}
        try:
            orders["stop"] = await self.ex.create_order(
                sym, "STOP_MARKET", opp, qty, None,
                {"stopPrice": float(self.ex.price_to_precision(sym, plan.stop_loss)), "reduceOnly": True})
            left = qty
            for i, (tp, w) in enumerate(zip(plan.tps, plan.tp_split)):
                q = left if i == len(plan.tps) - 1 else float(self.ex.amount_to_precision(sym, qty * w))
                left -= q
                orders[f"tp{i + 1}"] = await self.ex.create_order(
                    sym, "limit", opp, q, float(self.ex.price_to_precision(sym, tp)), {"reduceOnly": True})
        except Exception:
            # Never leave an unprotected position: flatten immediately, then re-raise.
            await self.ex.create_order(sym, "market", opp, qty, None, {"reduceOnly": True})
            log_event(log, logging.ERROR, "protective_orders_failed_position_flattened", symbol=ev.symbol)
            raise
        log_event(log, logging.INFO, "live_bracket_placed", symbol=ev.symbol, direction=d.value, qty=qty)
        return orders

    async def close(self) -> None:
        await self.ex.close()
