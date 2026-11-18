"""Read-only Alpaca paper-account check. Never submits, cancels, or closes orders."""

import os
from pathlib import Path

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")

    if not key or not secret:
        raise SystemExit(
            "Missing Alpaca credentials in .env; values were not displayed."
        )

    # Explicitly request the paper-trading endpoint.
    client = TradingClient(key, secret, paper=True)

    account = client.get_account()

    print("ALPACA PAPER ACCOUNT — READ ONLY")
    print("Paper mode requested: True")
    print(f"Account status: {account.status}")
    print(f"Trading blocked: {account.trading_blocked}")
    print(f"Account equity: ${float(account.equity):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Buying power: ${float(account.buying_power):,.2f}")
    print(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
    print()

    positions = client.get_all_positions()
    print(f"Open positions: {len(positions)}")
    for position in positions:
        print(
            f"  {position.symbol}: qty={position.qty}, "
            f"market_value=${float(position.market_value):,.2f}, "
            f"unrealized_pl=${float(position.unrealized_pl):,.2f}"
        )
    print()

    orders = client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
    )

    print(f"Open orders returned: {len(orders)}")
    for order in orders:
        print(
            f"  symbol={order.symbol}, side={order.side}, "
            f"qty={order.qty}, status={order.status}, "
            f"type={order.type}"
        )

    print()
    print("Read-only check complete. No orders were submitted or modified.")


if __name__ == "__main__":
    main()
