"""Download / update derivatives history: funding, basis, open interest & positioning, DVOL.

    python scripts/download_derivatives.py              # everything, incremental
    python scripts/download_derivatives.py --only dvol

First run takes ~15-30 minutes (about 2000 daily metrics archives); later runs only fetch new data.
"""

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.data.derivatives import sources  # noqa: E402
from app.data.derivatives.store import DATASETS, DerivativesStore  # noqa: E402

# Earliest public history of each source
STARTS = {
    "funding": datetime(2019, 9, 1, tzinfo=timezone.utc),
    "premium": datetime(2019, 12, 1, tzinfo=timezone.utc),
    "metrics": datetime(2020, 9, 1, tzinfo=timezone.utc),
    "dvol": datetime(2021, 3, 24, tzinfo=timezone.utc),
}


def ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--only", choices=DATASETS, nargs="*", help="datasets to update (default: all)")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    store = DerivativesStore(data_dir)
    symbol = args.symbol.upper()
    now = datetime.now(timezone.utc)

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for dataset in args.only or DATASETS:
            last = store.last_available(dataset, symbol)
            start = STARTS[dataset] if last is None else last.to_pydatetime() - timedelta(hours=2)
            print(f"{dataset:<8}: from {start:%Y-%m-%d %H:%M}", flush=True)

            if dataset == "funding":
                frame = sources.fetch_funding(client, symbol, ms(start))
            elif dataset == "premium":
                frame = sources.fetch_premium(client, symbol, ms(start))
                # Drop the still-open hourly candle
                frame = frame[frame["available_at"] <= now]
            elif dataset == "dvol":
                frame = sources.fetch_dvol(client, symbol[:3], ms(start), ms(now))
                frame = frame[frame["available_at"] <= now]
            else:
                # Daily archives appear with ~1 day delay
                frame = sources.fetch_metrics(
                    client, symbol, start.date(), (now - timedelta(days=1)).date(),
                    progress=lambda d: print(f"          metrics {d:%Y-%m}", flush=True),
                )

            merged = store.append(dataset, symbol, frame)
            print(
                f"          +{len(frame)} rows -> {len(merged)} total, "
                f"{merged['available_at'].min():%Y-%m-%d} .. {merged['available_at'].max():%Y-%m-%d %H:%M}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
