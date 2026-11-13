"""Build the timestamp-aligned v2 direction dataset.

The input parquet must contain ``timestamp``, ``funding_rate`` and
``basis_mean``. It may contain additional derivative columns. Matching is
causal: for each closed OHLCV candle, only the latest derivative observation at
or before that candle is used.

Example:
    python scripts/build_v2_features.py --source data/derivatives/BTCUSDT.parquet --horizon 24h
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.data.intervals import INTERVAL_MS  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402
from app.features.feature_pipeline import build_features  # noqa: E402
from app.features.targets import future_direction, future_return, horizon_to_periods  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="historical derivatives parquet")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS))
    parser.add_argument("--horizon", default="24h")
    args = parser.parse_args(argv)

    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    provider = settings.market_data_provider
    symbol = args.symbol.upper()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    ohlcv = MarketDataStorage(data_dir).load_processed(provider, symbol, args.interval)
    if ohlcv is None:
        print("No processed OHLCV data. Run scripts/download_data.py first.", file=sys.stderr)
        return 1
    if not source.exists():
        print(f"Derivative source does not exist: {source}", file=sys.stderr)
        return 1

    derivative = pd.read_parquet(source)
    required = {"timestamp", "funding_rate", "basis_mean"}
    missing = sorted(required - set(derivative.columns))
    if missing:
        print(f"Derivative source is missing columns: {', '.join(missing)}", file=sys.stderr)
        return 1
    ohlcv = ohlcv.copy()
    derivative = derivative.copy()
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
    derivative["timestamp"] = pd.to_datetime(derivative["timestamp"], utc=True)
    ohlcv = ohlcv.sort_values("timestamp")
    derivative = derivative.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    base = build_features(ohlcv, args.interval)
    derivative_columns = [column for column in derivative.columns if column != "timestamp"]
    merged = pd.merge_asof(
        base.sort_values("timestamp"),
        derivative[["timestamp", *derivative_columns]],
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(milliseconds=INTERVAL_MS[args.interval]),
    )
    periods = horizon_to_periods(args.horizon, args.interval)
    merged[f"target_return_{args.horizon}"] = future_return(merged["close"], periods)
    merged[f"target_direction_{args.horizon}"] = future_direction(merged[f"target_return_{args.horizon}"])
    output = data_dir / "features" / provider / f"{symbol}_{args.interval}_h{args.horizon}_v2.parquet"
    meta_path = output.with_name(output.stem + ".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    meta = {
        "feature_set": "v2_funding_basis",
        "symbol": symbol,
        "interval": args.interval,
        "horizon": args.horizon,
        "source_file": str(source),
        "rows": len(merged),
        "coverage_rows": int(merged["funding_rate"].notna().sum()),
        "first_timestamp": merged["timestamp"].min().isoformat(),
        "last_timestamp": merged["timestamp"].max().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved v2 dataset: {output}")
    print(f"Derivative-covered rows: {meta['coverage_rows']} / {meta['rows']}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
