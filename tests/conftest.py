import os

# Tests must never start the background paper-trading loop (it would call the network).
os.environ["PAPER_TRADING_LOOP"] = "false"
