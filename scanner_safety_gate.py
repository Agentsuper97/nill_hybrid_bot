import csv

from alpaca_safety_adapter import build_trade_proposal_from_snapshot
from alpaca_account_snapshot import get_account_snapshot
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os
from paper_safety import check_trade





def main():
    print("SCANNER → SAFETY GATE")
    print("READ-ONLY: no orders will be submitted.")
    print()

    load_dotenv()
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing Alpaca credentials in .env; values were not displayed.")

    client = TradingClient(key, secret, paper=True)
    snapshot = get_account_snapshot(client)

    print(f"Alpaca snapshot: {snapshot.get("status")}")
    print(f"Positions verified: {snapshot.get("positions_verified")}")
    print(f"Open orders verified: {snapshot.get("open_orders_verified")}")
    print(f"Daily P&L status: {snapshot.get("daily_pnl_status")}")
    print()

    if snapshot.get("status") != "SNAPSHOT VERIFIED — INFORMATIONAL":
        print("Safety gate stopped: Alpaca snapshot is not fully verified.")
        return

    with open("scanner_candidates.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = [
        row for row in rows
        if row.get("signal") == "POTENTIAL LONG"
    ]

    if not candidates:
        print("No scanner candidates.")
        return

    for row in candidates:
        symbol = row["symbol"]
        entry = float(row["close"])
        stop = float(row["reference_stop"])

        proposal = build_trade_proposal_from_snapshot(
            snapshot,
            symbol=symbol,
            entry_price=entry,
            stop_price=stop,
            pending_entry_symbols=[],
            strategy_daily_pnl=0.0,
            can_exit_before_close=True,
        )

        decision = check_trade(proposal)

        print(f"Candidate: {symbol}")
        print(f"Reference entry: ${entry:.2f}")
        print(f"Reference stop:  ${stop:.2f}")
        print(f"Safety decision: {'APPROVED' if decision.approved else 'BLOCKED'}")
        print(f"Reason: {decision.reason}")

        if decision.approved:
            print(f"Hypothetical quantity: {decision.quantity}")
            print(f"Planned risk: ${decision.planned_risk:.2f}")

        print()

    print("SAFETY GATE COMPLETE.")
    print("No orders were submitted, cancelled, or modified.")


if __name__ == "__main__":
    main()
