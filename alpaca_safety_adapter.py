"""Read-only adapter from a verified Alpaca snapshot to TradeProposal.

This module never submits, cancels, or modifies orders.
It deliberately keeps strategy sizing/daily P&L separate from the
broker account's $100k paper balance.
"""

from math import isfinite
from typing import Iterable

import paper_safety_config as config
from paper_safety import TradeProposal


VERIFIED_STATUS = "SNAPSHOT VERIFIED — INFORMATIONAL"


def build_trade_proposal_from_snapshot(
    snapshot: dict,
    *,
    symbol: str,
    entry_price: float,
    stop_price: float,
    pending_entry_symbols: Iterable[str],
    strategy_daily_pnl: float,
    can_exit_before_close: bool,
) -> TradeProposal:
    """Build a TradeProposal only from a fully verified read-only snapshot.

    Broker account equity and broker daily P&L are intentionally NOT used
    for strategy sizing or the strategy daily-loss limit. The strategy
    remains based on config.STARTING_EQUITY ($10,000) unless the caller
    explicitly changes the strategy configuration.

    No order capability exists here.
    """

    if not isinstance(snapshot, dict):
        raise ValueError("Invalid Alpaca snapshot.")

    if snapshot.get("status") != VERIFIED_STATUS:
        raise ValueError("Alpaca snapshot is not fully verified.")

    if snapshot.get("positions_verified") is not True:
        raise ValueError("Alpaca positions are not verified.")

    if snapshot.get("open_orders_verified") is not True:
        raise ValueError("Alpaca open orders are not verified.")

    if snapshot.get("daily_pnl_status") != "ALLOWED":
        raise ValueError("Alpaca daily P&L status is not allowed.")

    if snapshot.get("account_status") != "ACTIVE":
        raise ValueError("Alpaca account is not active.")

    if snapshot.get("trading_blocked") is not False:
        raise ValueError("Alpaca account is broker-blocked.")

    positions = snapshot.get("positions")
    if not isinstance(positions, list):
        raise ValueError("Invalid verified positions.")

    if not isinstance(pending_entry_symbols, (set, frozenset, list, tuple)):
        raise ValueError("Invalid pending-entry data.")

    if (
        not isinstance(strategy_daily_pnl, (int, float))
        or isinstance(strategy_daily_pnl, bool)
        or not isfinite(strategy_daily_pnl)
    ):
        raise ValueError("Invalid strategy daily P&L.")

    # Deliberately use strategy capital, NOT broker account equity.
    strategy_equity = config.STARTING_EQUITY

    if (
        not isinstance(strategy_equity, (int, float))
        or isinstance(strategy_equity, bool)
        or not isfinite(strategy_equity)
        or strategy_equity <= 0
    ):
        raise ValueError("Invalid strategy starting equity.")

    return TradeProposal(
        symbol=symbol,
        entry_price=entry_price,
        stop_price=stop_price,
        open_positions=len(positions),
        pending_entry_symbols=frozenset(pending_entry_symbols),
        daily_pnl=strategy_daily_pnl,
        paper_mode=True,
        current_equity=strategy_equity,
        can_exit_before_close=can_exit_before_close,
    )
