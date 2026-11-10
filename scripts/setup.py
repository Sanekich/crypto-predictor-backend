"""Prepare everything the API needs on a fresh machine (data and models are not stored in Git).

Run from the repository root with the virtual environment's Python:

    python scripts/setup.py                 # download/update data, build features, train missing models
    python scripts/setup.py --retrain       # retrain models even if they already exist

Steps for each horizon (24h and 7d by default):
    1. download or update BTCUSDT 1h candles from Binance
    2. build the feature dataset
    3. train the range model (skipped if one already exists, unless --retrain)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_features  # noqa: E402
import download_data  # noqa: E402
import train_range_model  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry  # noqa: E402
from app.services.prediction_service import RANGE_MODEL_TYPE  # noqa: E402

DEFAULT_HORIZONS = ["24h", "7d"]


def step(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizons", nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--retrain", action="store_true", help="train new models even if they exist")
    args = parser.parse_args(argv)

    step(f"1/3  Market data: {args.symbol} {args.interval}")
    if download_data.main(["--symbol", args.symbol, "--interval", args.interval]) != 0:
        print("Data download failed. Check the internet connection and try again.", file=sys.stderr)
        return 1

    settings = get_settings()
    artifacts_dir = (
        settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    )
    registry = ModelRegistry(artifacts_dir)

    for horizon in args.horizons:
        step(f"2/3  Features: horizon {horizon}")
        if build_features.main(["--symbol", args.symbol, "--interval", args.interval, "--horizon", horizon]) != 0:
            return 1

        step(f"3/3  Range model: horizon {horizon}")
        existing = registry.latest(RANGE_MODEL_TYPE, args.symbol, args.interval, horizon)
        if existing and not args.retrain:
            print(f"Model already exists: {existing['model_id']} (use --retrain to train a new one)")
            continue
        if train_range_model.main(["--symbol", args.symbol, "--interval", args.interval, "--horizon", horizon]) != 0:
            return 1

    step("Done. Start the API with:")
    print("    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
