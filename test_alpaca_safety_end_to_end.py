import pytest

from alpaca_safety_adapter import build_trade_proposal_from_snapshot
from paper_safety import check_trade


VERIFIED_SNAPSHOT = {
    "status": "SNAPSHOT VERIFIED — INFORMATIONAL",
    "account_status": "ACTIVE",
    "trading_blocked": False,
    "daily_pnl_status": "ALLOWED",
    "positions_verified": True,
    "open_orders_verified": True,
    "positions": [],
    "open_orders": [],
}


def make_proposal(**overrides):
    values = {
        "snapshot": VERIFIED_SNAPSHOT,
        "symbol": "AAPL",
        "entry_price": 100.00,
        "stop_price": 99.00,
        "pending_entry_symbols": set(),
        "strategy_daily_pnl": 0.00,
        "can_exit_before_close": True,
    }
    values.update(overrides)
    return build_trade_proposal_from_snapshot(**values)


def test_verified_snapshot_allows_safe_proposal():
    proposal = make_proposal()
    decision = check_trade(proposal)

    assert decision.approved is True
    assert decision.quantity == 50
    assert decision.planned_risk == 50.00


def test_daily_loss_limit_blocks():
    proposal = make_proposal(strategy_daily_pnl=-200.00)
    decision = check_trade(proposal)

    assert decision.approved is False
    assert "Daily loss limit" in decision.reason


def test_max_positions_blocks():
    snapshot = {**VERIFIED_SNAPSHOT, "positions": [{}, {}, {}, {}, {}]}
    proposal = make_proposal(snapshot=snapshot)
    decision = check_trade(proposal)

    assert decision.approved is False
    assert "Maximum open positions" in decision.reason


def test_duplicate_pending_entry_blocks():
    proposal = make_proposal(pending_entry_symbols={"AAPL"})
    decision = check_trade(proposal)

    assert decision.approved is False
    assert "already pending" in decision.reason


def test_cannot_exit_before_close_blocks():
    proposal = make_proposal(can_exit_before_close=False)
    decision = check_trade(proposal)

    assert decision.approved is False
    assert "exit before market close" in decision.reason


def test_unknown_snapshot_blocks():
    snapshot = {
        **VERIFIED_SNAPSHOT,
        "status": "UNKNOWN — BLOCKED",
        "daily_pnl_status": "UNKNOWN — BLOCKED",
        "positions_verified": False,
        "open_orders_verified": False,
    }

    with pytest.raises(ValueError, match="not fully verified"):
        make_proposal(snapshot=snapshot)


def test_broker_block_blocks():
    snapshot = {**VERIFIED_SNAPSHOT, "trading_blocked": True}

    with pytest.raises(ValueError, match="broker-blocked"):
        make_proposal(snapshot=snapshot)
