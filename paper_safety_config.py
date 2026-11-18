# Initial safeguards for the Alpaca paper-trading project.
# Configuration only: this file does not submit orders.

PAPER_ONLY = True
STARTING_EQUITY = 10_000.00
RISK_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 5
MAX_DAILY_LOSS = 0.02
ALLOW_OVERNIGHT_POSITIONS = False
BLOCK_DUPLICATE_ENTRIES = True
