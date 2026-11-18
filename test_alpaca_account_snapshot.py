import unittest
from types import SimpleNamespace
from unittest.mock import patch

from alpaca_account_snapshot import get_account_snapshot


def make_account(status="ACTIVE", trading_blocked=False, equity="100000"):
    return SimpleNamespace(
        status=status,
        trading_blocked=trading_blocked,
        equity=equity,
    )


def make_position(symbol="AAPL", qty="3", side="long"):
    return SimpleNamespace(symbol=symbol, qty=qty, side=side)


def make_order(symbol="MSFT", side="buy", order_type="limit", status="new"):
    return SimpleNamespace(
        symbol=symbol,
        side=side,
        type=order_type,
        status=status,
    )


class FakeClient:
    def __init__(
        self,
        account=None,
        positions=None,
        orders=None,
        history=None,
    ):
        self.account = account or make_account()
        self.positions = [] if positions is None else positions
        self.orders = [] if orders is None else orders
        self.history = history or SimpleNamespace(
            timestamp=[1],
            profit_loss=[10.0],
        )
        self.calls = []

    def get_account(self):
        self.calls.append("get_account")
        return self.account

    def get_portfolio_history(self, request):
        self.calls.append("get_portfolio_history")
        return self.history

    def get_all_positions(self):
        self.calls.append("get_all_positions")
        return self.positions

    def get_orders(self, request):
        self.calls.append("get_orders")
        return self.orders


class AccountSnapshotTests(unittest.TestCase):

    @patch(
        "alpaca_account_snapshot.evaluate_daily_loss",
        return_value=("ALLOWED", "Daily loss limit not reached.", 10.0, 5.0),
    )
    def test_complete_snapshot_is_informational(self, _check):
        client = FakeClient(
            positions=[make_position()],
            orders=[make_order()],
        )

        result = get_account_snapshot(client)

        self.assertEqual(
            result["status"],
            "SNAPSHOT VERIFIED — INFORMATIONAL",
        )
        self.assertTrue(result["positions_verified"])
        self.assertTrue(result["open_orders_verified"])
        self.assertEqual(result["positions"][0]["symbol"], "AAPL")
        self.assertEqual(result["open_orders"][0]["symbol"], "MSFT")
        self.assertEqual(
            client.calls,
            [
                "get_account",
                "get_portfolio_history",
                "get_all_positions",
                "get_orders",
            ],
        )

    @patch(
        "alpaca_account_snapshot.evaluate_daily_loss",
        return_value=(
            "UNKNOWN — BLOCKED",
            "Daily P&L could not be verified.",
            None,
            None,
        ),
    )
    def test_unknown_daily_pnl_still_retrieves_positions_and_orders(
        self, _check
    ):
        client = FakeClient(
            positions=[make_position()],
            orders=[make_order()],
        )

        result = get_account_snapshot(client)

        self.assertEqual(result["status"], "UNKNOWN — BLOCKED")
        self.assertTrue(result["positions_verified"])
        self.assertTrue(result["open_orders_verified"])
        self.assertEqual(len(result["positions"]), 1)
        self.assertEqual(len(result["open_orders"]), 1)

    @patch(
        "alpaca_account_snapshot.evaluate_daily_loss",
        return_value=("BLOCKED", "Daily loss limit reached.", -200.0, 5.0),
    )
    def test_daily_loss_block_still_retrieves_positions_and_orders(
        self, _check
    ):
        client = FakeClient(
            positions=[make_position()],
            orders=[make_order()],
        )

        result = get_account_snapshot(client)

        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(result["positions_verified"])
        self.assertTrue(result["open_orders_verified"])

    def test_inactive_account_does_not_claim_positions_or_orders_verified(self):
        client = FakeClient(account=make_account(status="INACTIVE"))

        result = get_account_snapshot(client)

        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["positions_verified"])
        self.assertFalse(result["open_orders_verified"])
        self.assertNotIn("get_all_positions", client.calls)
        self.assertNotIn("get_orders", client.calls)

    def test_invalid_equity_fails_closed(self):
        client = FakeClient(account=make_account(equity="nan"))

        result = get_account_snapshot(client)

        self.assertEqual(result["status"], "UNKNOWN — BLOCKED")
        self.assertFalse(result["positions_verified"])
        self.assertFalse(result["open_orders_verified"])

    @patch(
        "alpaca_account_snapshot.evaluate_daily_loss",
        return_value=("ALLOWED", "Daily loss limit not reached.", 10.0, 5.0),
    )
    def test_invalid_positions_response_fails_closed(self, _check):
        client = FakeClient(positions=None)
        client.get_all_positions = lambda: {"unexpected": "response"}

        result = get_account_snapshot(client)

        self.assertEqual(result["status"], "UNKNOWN — BLOCKED")
        self.assertFalse(result["positions_verified"])
        self.assertFalse(result["open_orders_verified"])

    @patch(
        "alpaca_account_snapshot.evaluate_daily_loss",
        return_value=("ALLOWED", "Daily loss limit not reached.", 10.0, 5.0),
    )
    def test_invalid_open_order_fails_closed(self, _check):
        client = FakeClient(orders=[make_order(side="unexpected")])

        result = get_account_snapshot(client)

        self.assertEqual(result["status"], "UNKNOWN — BLOCKED")
        self.assertTrue(result["positions_verified"])
        self.assertFalse(result["open_orders_verified"])

    @patch(
        "alpaca_account_snapshot.evaluate_daily_loss",
        return_value=("ALLOWED", "Daily loss limit not reached.", 10.0, 5.0),
    )
    def test_500_open_orders_fails_closed(self, _check):
        client = FakeClient(
            orders=[make_order() for _ in range(500)]
        )

        result = get_account_snapshot(client)

        self.assertEqual(result["status"], "UNKNOWN — BLOCKED")
        self.assertTrue(result["positions_verified"])
        self.assertFalse(result["open_orders_verified"])
        self.assertIn("500-order limit", result["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
