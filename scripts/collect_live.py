"""Record Binance real-time data (order book, trade flow, funding, liquidations, open interest).

This data has no downloadable history: it exists only if it is recorded
while it happens. Keep the collector running as long as possible; every
hour it is off is a gap in the dataset.

Run from the repository root (stop with Ctrl+C, buffers are flushed):

    python scripts/collect_live.py
    python scripts/collect_live.py --symbol ETHUSDT

Output: data/live/<dataset>/<SYMBOL>/<YYYY-MM-DD>/<HH>.parquet (one row per minute)
Status: data/live/_status_<SYMBOL>.json
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.data.live.collector import LiveCollector  # noqa: E402
from app.data.live.writer import LiveDataWriter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--flush-seconds", type=float, default=60.0, help="how often buffers are written to disk")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    writer = LiveDataWriter(data_dir / "live", args.symbol)
    collector = LiveCollector(writer, args.symbol, flush_interval=args.flush_seconds)

    print(f"Collecting {args.symbol} live data into {writer.live_dir} (Ctrl+C to stop)")
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        pass
    print("Stopped. Data saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
