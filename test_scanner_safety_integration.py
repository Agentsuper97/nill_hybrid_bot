from alpaca_safety_adapter import build_trade_proposal_from_snapshot
from paper_safety import check_trade


SNAPSHOT = {
    "status": "SNAPSHOT VERIFIED — INFORMATIONAL",
    "positions_verified": True,
    "open_orders_verified": True,
    "daily_pnl_status": "ALLOWED",
    "account_status": "ACTIVE",
    "trading_blocked": False,
    "positions": [],
}


def make_proposal(**overrides):
    values = {
        "symbol": "ORCL",
        "entry_price": 116.07,
        "stop_price": 113.90,
        "pending_entry_symbols": [],
        "strategy_daily_pnl": 0.0,
        "can_exit_before_close": True,
    }
    values.update(overrides)

    return build_trade_proposal_from_snapshot(
        SNAPSHOT,
        **values,
    )


print("SCANNER → SAFETY INTEGRATION TEST")
print("Read-only test: no orders will be submitted.")
print()

# 1. Normal hypothetical candidate
decision = check_trade(make_proposal())

assert decision.approved is True
assert decision.quantity > 0
print("PASS: ORCL hypothetical candidate passed safety checks.")
print(f"      Hypothetical quantity: {decision.quantity}")
print(f"      Planned risk: ${decision.planned_risk:.2f}")

# 2. Daily-loss protection
decision = check_trade(make_proposal(strategy_daily_pnl=-200.0))

assert decision.approved is False
assert "Daily loss limit" in decision.reason
print("PASS: $200 daily-loss limit blocks new entries.")

# 3. Exit-before-close protection
decision = check_trade(make_proposal(can_exit_before_close=False))

assert decision.approved is False
assert "exit before market close" in decision.reason
print("PASS: cannot-exit-before-close protection blocks entry.")

print()
print("All scanner → safety tests passed.")
print("No orders were submitted, cancelled, or modified.")
