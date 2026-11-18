"""Offline tests for paper_safety.py. No broker connection or orders."""

from paper_safety import TradeProposal, check_trade


def proposal(**overrides):
    values = {
        "symbol": "AAPL",
        "entry_price": 100.00,
        "stop_price": 99.00,
        "open_positions": 0,
        "pending_entry_symbols": frozenset(),
        "daily_pnl": 0.00,
        "paper_mode": True,
    }
    values.update(overrides)
    return TradeProposal(**values)


def expect_blocked(label, **overrides):
    decision = check_trade(proposal(**overrides))
    assert not decision.approved, (
        f"FAIL: {label} was incorrectly approved: {decision}"
    )
    print(f"PASS: blocked — {label}")


def main():
    # A $1 stop distance and $50 risk budget should allow 50 shares.
    decision = check_trade(proposal())
    assert decision.approved, f"Expected safe proposal to pass: {decision}"
    assert decision.quantity == 50, (
        f"Expected 50 shares, got {decision.quantity}"
    )
    assert decision.planned_risk == 50.00
    print("PASS: valid proposal; quantity and planned risk are correct")

    expect_blocked("paper mode is false", paper_mode=False)
    expect_blocked("empty symbol", symbol="   ")
    expect_blocked("maximum open positions", open_positions=5)
    expect_blocked(
        "duplicate pending entry",
        pending_entry_symbols=frozenset({"aapl"}),
    )
    expect_blocked("daily loss limit reached", daily_pnl=-200.00)
    expect_blocked("non-positive entry", entry_price=0)
    expect_blocked("stop at entry", stop_price=100.00)
    expect_blocked("stop above entry", stop_price=101.00)
    expect_blocked(
        "one share exceeds risk budget",
        entry_price=1000.00,
        stop_price=900.00,
    )



    # Additional tests for the revised safeguards.
    expect_blocked(
        "cannot confirm exit before close",
        can_exit_before_close=False,
    )
    expect_blocked(
        "NaN entry price",
        entry_price=float("nan"),
    )
    expect_blocked(
        "infinite stop price",
        stop_price=float("inf"),
    )

    # With $20,000 equity, 0.5% risk is $100.
    larger_account = check_trade(proposal(current_equity=20_000.00))
    assert larger_account.approved, larger_account
    assert larger_account.quantity == 100, (
        f"Expected 100 shares, got {larger_account.quantity}"
    )
    assert larger_account.planned_risk == 100.00
    print("PASS: position sizing uses supplied current equity")

    # Negative equity must never be accepted.
    expect_blocked("non-positive current equity", current_equity=0)

    # Regression tests: malformed pending-entry data must be rejected.
    expect_blocked(
        "pending-entry data contains a non-string",
        pending_entry_symbols=frozenset({123}),
    )
    expect_blocked(
        "pending-entry data is missing",
        pending_entry_symbols=None,
    )

    # Regression test: invalid position-limit configuration must be
    # rejected explicitly, not merely block trades by accident.
    import paper_safety_config as config

    original_max_positions = config.MAX_OPEN_POSITIONS
    try:
        config.MAX_OPEN_POSITIONS = 0
        decision = check_trade(proposal())
        assert not decision.approved, (
            f"FAIL: invalid maximum-position configuration was approved: {decision}"
        )
        assert decision.reason == "Invalid maximum-position configuration.", (
            "FAIL: expected explicit configuration rejection, got: "
            f"{decision.reason}"
        )
        print("PASS: blocked — invalid maximum-position configuration")
    finally:
        config.MAX_OPEN_POSITIONS = original_max_positions

    # Regression test: Boolean risk settings must be rejected explicitly.
    original_risk = config.RISK_PER_TRADE
    try:
        config.RISK_PER_TRADE = True
        decision = check_trade(proposal())
        assert not decision.approved, (
            f"FAIL: Boolean risk configuration was approved: {decision}"
        )
        assert decision.reason == "Invalid risk configuration.", (
            "FAIL: expected explicit risk-configuration rejection, got: "
            f"{decision.reason}"
        )
        print("PASS: blocked — Boolean risk configuration")
    finally:
        config.RISK_PER_TRADE = original_risk

    # Regression test: Boolean daily-loss settings must be rejected explicitly.
    original_daily_loss = config.MAX_DAILY_LOSS
    try:
        config.MAX_DAILY_LOSS = True
        decision = check_trade(proposal())
        assert not decision.approved, (
            f"FAIL: Boolean daily-loss configuration was approved: {decision}"
        )
        assert decision.reason == "Invalid daily-loss configuration.", (
            "FAIL: expected explicit daily-loss rejection, got: "
            f"{decision.reason}"
        )
        print("PASS: blocked — Boolean daily-loss configuration")
    finally:
        config.MAX_DAILY_LOSS = original_daily_loss

    # Regression test: PAPER_ONLY must be an actual Boolean True.
    original_paper_only = config.PAPER_ONLY
    try:
        config.PAPER_ONLY = 1
        decision = check_trade(proposal())
        assert not decision.approved, "FAIL: non-Boolean PAPER_ONLY was approved."
        assert decision.reason == "Paper-only verification failed."
        print("PASS: blocked — non-Boolean PAPER_ONLY configuration")
    finally:
        config.PAPER_ONLY = original_paper_only

    # Regression test: overnight setting must be an actual Boolean.
    original_overnight = config.ALLOW_OVERNIGHT_POSITIONS
    try:
        config.ALLOW_OVERNIGHT_POSITIONS = 0
        decision = check_trade(proposal())
        assert not decision.approved, "FAIL: non-Boolean overnight setting was approved."
        assert decision.reason == "Invalid Boolean safety configuration."
        print("PASS: blocked — non-Boolean overnight configuration")
    finally:
        config.ALLOW_OVERNIGHT_POSITIONS = original_overnight

    # Regression test: duplicate-entry setting must be an actual Boolean.
    original_duplicates = config.BLOCK_DUPLICATE_ENTRIES
    try:
        config.BLOCK_DUPLICATE_ENTRIES = None
        decision = check_trade(proposal())
        assert not decision.approved, "FAIL: non-Boolean duplicate setting was approved."
        assert decision.reason == "Invalid Boolean safety configuration."
        print("PASS: blocked — non-Boolean duplicate-entry configuration")
    finally:
        config.BLOCK_DUPLICATE_ENTRIES = original_duplicates

    print("All offline safety tests passed, including additional safeguards.")


if __name__ == "__main__":
    main()
