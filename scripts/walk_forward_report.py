"""Evaluate a direction model across expanding out-of-sample periods.

Run from the backend repository root:
    python scripts/walk_forward_report.py --horizon 24h
    python scripts/walk_forward_report.py --horizon 7d

This command is read-only: it trains temporary fold models and does not change
saved production artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.features.feature_sets import load_direction_dataset  # noqa: E402
from app.ml.evaluation.walk_forward import evaluate_direction_walk_forward  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizon", default="24h")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--feature-set", choices=("v1", "v2_funding_basis", "v3_open_interest"), default="v1")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    try:
        dataset = load_direction_dataset(data_dir, settings.market_data_provider, args.symbol, args.interval, args.horizon, args.feature_set)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Cannot prepare {args.feature_set} dataset: {exc}", file=sys.stderr)
        return 1
    results = evaluate_direction_walk_forward(
        dataset.frame,
        dataset.feature_columns,
        dataset.target_direction_column,
        dataset.horizon_periods,
        folds=args.folds,
    )
    if not results:
        print("No valid folds could be created.", file=sys.stderr)
        return 1

    print(f"Walk-forward direction report: {args.symbol} {args.interval} {args.horizon} ({args.feature_set})")
    print(f"{'fold':>4} {'test period':^45} {'accuracy':>10} {'indep acc':>10} {'brier':>10} {'max conf':>10}")
    for result in results:
        period = f"{result['test_start'][:10]} .. {result['test_end'][:10]}"
        print(
            f"{result['fold']:>4} {period:^45} {result['accuracy']:>9.1%}"
            f" {result['independent_accuracy']:>9.1%} {result['brier']:>10.4f} {result['max_confidence']:>9.1%}"
        )
    accuracies = [item["accuracy"] for item in results]
    independent_accuracies = [item["independent_accuracy"] for item in results]
    print(f"mean accuracy: {sum(accuracies) / len(accuracies):.1%}")
    print(f"worst accuracy: {min(accuracies):.1%}")
    print(f"independent mean accuracy: {sum(independent_accuracies) / len(independent_accuracies):.1%}")
    print(f"independent worst accuracy: {min(independent_accuracies):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
