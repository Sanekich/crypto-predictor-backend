"""Download historical OHLCV candles from Binance and build a validated dataset.

Run from the repository root with the virtual environment active:

    python scripts/download_data.py                      # BTCUSDT 1h, full history / update
    python scripts/download_data.py --interval 4h
    python scripts/download_data.py --symbol ETHUSDT --start 2020-01-01
    python scripts/download_data.py --full-refresh       # ignore saved data, download again
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.data.collection.binance_client import BinanceAPIError, BinanceClient  # noqa: E402
from app.data.collection.collector import MarketDataCollector  # noqa: E402
from app.data.intervals import INTERVAL_MS  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402

BTCUSDT_LISTING_DATE = "2017-08-17"


def parse_utc_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT", help="trading pair, e.g. BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS), help="candle interval")
    parser.add_argument("--start", default=BTCUSDT_LISTING_DATE, help="start date (UTC), e.g. 2020-01-01")
    parser.add_argument("--full-refresh", action="store_true", help="re-download all data from --start")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)

    if settings.market_data_provider != "binance":
        print(f"Unsupported MARKET_DATA_PROVIDER: {settings.market_data_provider}", file=sys.stderr)
        return 2

    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    storage = MarketDataStorage(data_dir)

    try:
        with BinanceClient(base_url=settings.binance_base_url) as client:
            result = MarketDataCollector(client, storage).collect(
                symbol=args.symbol,
                interval=args.interval,
                start=parse_utc_date(args.start),
                full_refresh=args.full_refresh,
            )
    except BinanceAPIError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    report = result.report
    print()
    print(f"Symbol / interval : {result.symbol} {result.interval}")
    print(f"Fetched candles   : {result.fetched_rows}")
    print(f"Closed candles    : {result.processed_rows}")
    print(f"Period (UTC)      : {report.first_timestamp} .. {report.last_timestamp}")
    print(f"Gaps              : {len(report.gaps)} ({report.missing_candles} missing candles)")
    print(f"Valid             : {'yes' if report.is_valid else 'NO'}")
    for error in report.errors:
        print(f"  ERROR: {error}")
    print(f"Raw file          : {result.raw_path}")
    print(f"Processed file    : {result.processed_path or '(not updated: validation failed)'}")
    print(f"Quality report    : {result.report_path}")

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
