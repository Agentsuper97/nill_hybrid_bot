import unittest

import paper_safety_config as config
from alpaca_safety_adapter import build_trade_proposal_from_snapshot


def verified_snapshot(
    *,
    status="SNAPSHOT VERIFIED — INFORMATIONAL",
    positions_verified=True,
    open_orders_verified=True,
    daily_pnl_status="ALLOWED",
    account_status="ACTIVE",
    trading_blocked=False,
    positions=None,
):
    return {
        "status": status,
        "positions_verified": positions_verified,
        "open_orders_verified": open_orders_verified,
        "daily_pnl_status": daily_pnl_status,
        "account_status": account_status,
        "trading_blocked": trading_blocked,
        "positions": [] if positions is None else positions,
        # Deliberately different from the $10k strategy capital.
        "account_equity": 100_000.00,
    }


class AlpacaSafetyAdapterTests(unittest.TestCase):
    def test_verified_snapshot_builds_proposal(self):
        snapshot = verified_snapshot(
            positions=[{"symbol": "NVDA"}, {"symbol": "AMD"}]
        )

        proposal = build_trade_proposal_from_snapshot(
            snapshot,
            symbol="MSFT",
            entry_price=500.00,
            stop_price=495.00,
            pending_entry_symbols={"TSLA"},
            strategy_daily_pnl=-25.00,
            can_exit_before_close=True,
        )

        self.assertEqual(proposal.symbol, "MSFT")
        self.assertEqual(proposal.open_positions, 2)
        self.assertEqual(proposal.pending_entry_symbols, frozenset({"TSLA"}))
        self.assertEqual(proposal.daily_pnl, -25.00)
        self.assertTrue(proposal.paper_mode)
        self.assertEqual(proposal.current_equity, config.STARTING_EQUITY)

    def test_broker_equity_is_not_used_for_strategy_sizing(self):
        snapshot = verified_snapshot()

        proposal = build_trade_proposal_from_snapshot(
            snapshot,
            symbol="NVDA",
            entry_price=200.00,
            stop_price=190.00,
            pending_entry_symbols=set(),
            strategy_daily_pnl=0.00,
            can_exit_before_close=True,
        )

        self.assertEqual(proposal.current_equity, 10_000.00)
        self.assertNotEqual(proposal.current_equity, snapshot["account_equity"])

    def test_unverified_snapshot_is_rejected(self):
        snapshot = verified_snapshot(status="BLOCKED")

        with self.assertRaises(ValueError):
            build_trade_proposal_from_snapshot(
                snapshot,
                symbol="NVDA",
                entry_price=200.00,
                stop_price=190.00,
                pending_entry_symbols=set(),
                strategy_daily_pnl=0.00,
                can_exit_before_close=True,
            )

    def test_unverified_positions_are_rejected(self):
        snapshot = verified_snapshot(positions_verified=False)

        with self.assertRaises(ValueError):
            build_trade_proposal_from_snapshot(
                snapshot,
                symbol="NVDA",
                entry_price=200.00,
                stop_price=190.00,
                pending_entry_symbols=set(),
                strategy_daily_pnl=0.00,
                can_exit_before_close=True,
            )

    def test_unverified_open_orders_are_rejected(self):
        snapshot = verified_snapshot(open_orders_verified=False)

        with self.assertRaises(ValueError):
            build_trade_proposal_from_snapshot(
                snapshot,
                symbol="NVDA",
                entry_price=200.00,
                stop_price=190.00,
                pending_entry_symbols=set(),
                strategy_daily_pnl=0.00,
                can_exit_before_close=True,
            )

    def test_blocked_daily_pnl_is_rejected(self):
        snapshot = verified_snapshot(daily_pnl_status="BLOCKED")

        with self.assertRaises(ValueError):
            build_trade_proposal_from_snapshot(
                snapshot,
                symbol="NVDA",
                entry_price=200.00,
                stop_price=190.00,
                pending_entry_symbols=set(),
                strategy_daily_pnl=0.00,
                can_exit_before_close=True,
            )

    def test_inactive_account_is_rejected(self):
        snapshot = verified_snapshot(account_status="INACTIVE")

        with self.assertRaises(ValueError):
            build_trade_proposal_from_snapshot(
                snapshot,
                symbol="NVDA",
                entry_price=200.00,
                stop_price=190.00,
                pending_entry_symbols=set(),
                strategy_daily_pnl=0.00,
                can_exit_before_close=True,
            )

    def test_broker_block_is_rejected(self):
        snapshot = verified_snapshot(trading_blocked=True)

        with self.assertRaises(ValueError):
            build_trade_proposal_from_snapshot(
                snapshot,
                symbol="NVDA",
                entry_price=200.00,
                stop_price=190.00,
                pending_entry_symbols=set(),
                strategy_daily_pnl=0.00,
                can_exit_before_close=True,
            )

    def test_invalid_strategy_daily_pnl_is_rejected(self):
        snapshot = verified_snapshot()

        with self.assertRaises(ValueError):
            build_trade_proposal_from_snapshot(
                snapshot,
                symbol="NVDA",
                entry_price=200.00,
                stop_price=190.00,
                pending_entry_symbols=set(),
                strategy_daily_pnl=float("nan"),
                can_exit_before_close=True,
            )


if __name__ == "__main__":
    unittest.main()


class AdapterToSafetyIntegrationTests(unittest.TestCase):
    def test_verified_snapshot_can_feed_existing_safety_check(self):
        from paper_safety import check_trade

        snapshot = verified_snapshot(
            positions=[{"symbol": "AMD"}, {"symbol": "NVDA"}]
        )

        proposal = build_trade_proposal_from_snapshot(
            snapshot,
            symbol="AAPL",
            entry_price=100.00,
            stop_price=99.00,
            pending_entry_symbols=set(),
            strategy_daily_pnl=0.00,
            can_exit_before_close=True,
        )

        decision = check_trade(proposal)

        self.assertTrue(decision.approved, decision)
        self.assertEqual(decision.quantity, 50)
        self.assertEqual(decision.planned_risk, 50.00)

    def test_verified_snapshot_preserves_max_position_safety(self):
        from paper_safety import check_trade

        snapshot = verified_snapshot(
            positions=[
                {"symbol": "AAPL"},
                {"symbol": "AMD"},
                {"symbol": "NVDA"},
                {"symbol": "MSFT"},
                {"symbol": "META"},
            ]
        )

        proposal = build_trade_proposal_from_snapshot(
            snapshot,
            symbol="TSLA",
            entry_price=100.00,
            stop_price=99.00,
            pending_entry_symbols=set(),
            strategy_daily_pnl=0.00,
            can_exit_before_close=True,
        )

        decision = check_trade(proposal)

        self.assertFalse(decision.approved)
        self.assertEqual(
            decision.reason,
            "Maximum open positions reached.",
        )

    def test_strategy_daily_loss_remains_separate_from_broker_snapshot(self):
        from paper_safety import check_trade

        snapshot = verified_snapshot()

        proposal = build_trade_proposal_from_snapshot(
            snapshot,
            symbol="AAPL",
            entry_price=100.00,
            stop_price=99.00,
            pending_entry_symbols=set(),
            strategy_daily_pnl=-200.00,
            can_exit_before_close=True,
        )

        decision = check_trade(proposal)

        self.assertFalse(decision.approved)
        self.assertEqual(
            decision.reason,
            "Daily loss limit reached; entries stopped.",
        )


if __name__ == "__main__":
    unittest.main()
