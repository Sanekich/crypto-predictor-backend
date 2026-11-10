"""Build the ML dataset (features + targets) from processed OHLCV data.

Run from the repository root after scripts/download_data.py:

    python scripts/build_features.py                         # BTCUSDT 1h, horizon 24h
    python scripts/build_features.py --horizon 4h
    python scripts/build_features.py --symbol ETHUSDT --interval 4h --horizon 1d
"""

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.data.intervals import INTERVAL_MS  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402
from app.features.feature_pipeline import build_dataset  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS))
    parser.add_argument("--horizon", default="24h", help="prediction horizon, e.g. 4h, 24h, 3d")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    provider = settings.market_data_provider
    symbol = args.symbol.upper()

    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    storage = MarketDataStorage(data_dir)
    source_path = storage.processed_path(provider, symbol, args.interval)
    ohlcv = storage.load_processed(provider, symbol, args.interval)
    if ohlcv is None:
        print(f"No processed data at {source_path}. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    try:
        dataset = build_dataset(ohlcv, args.interval, args.horizon)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    dataset.metadata.update(
        {
            "provider": provider,
            "symbol": symbol,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source_path.relative_to(data_dir)),
            "source_sha256": file_sha256(source_path),
        }
    )
    meta = dataset.describe()
    path, meta_path = storage.save_features(dataset.frame, meta, provider, symbol, args.interval, args.horizon)

    print()
    print(f"Symbol / interval : {symbol} {args.interval}, horizon {args.horizon} ({dataset.horizon_periods} candles)")
    print(f"Rows              : {meta['rows']} (dropped {meta['rows_dropped']}: warm-up, gaps, last {dataset.horizon_periods} rows without target)")
    print(f"Period (UTC)      : {meta['first_timestamp']} .. {meta['last_timestamp']}")
    print(f"Features          : {len(dataset.feature_columns)}")
    print(f"Positive targets  : {meta['positive_target_share']:.1%}")
    print(f"Dataset           : {path}")
    print(f"Metadata          : {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
