"""Train an experimental three-class direction model.

Labels are: 0=down, 1=neutral, 2=up. The model is not used by the API yet;
its artifacts are saved with status ``shadow`` for comparison with the binary
models.

Examples:
    python scripts/train_neutral_direction_model.py --horizon 24h
    python scripts/train_neutral_direction_model.py --horizon 7d
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.data.intervals import interval_to_timedelta  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION, build_dataset  # noqa: E402
from app.features.targets import future_direction_with_neutral  # noqa: E402
from app.ml.evaluation.metrics import wilson_lower_bound  # noqa: E402
from app.ml.evaluation.splits import chronological_split  # noqa: E402
from app.ml.models.neutral_direction_model import XGBoostNeutralDirectionModel  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry, make_model_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizon", default="24h")
    parser.add_argument("--neutral-return", type=float, default=0.001)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    artifacts_dir = settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    data = MarketDataStorage(data_dir).load_processed(settings.market_data_provider, args.symbol, args.interval)
    if data is None:
        print("No processed data. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    dataset = build_dataset(data, args.interval, args.horizon)
    future_ret = dataset.frame[dataset.target_return_column]
    labels = future_direction_with_neutral(future_ret, args.neutral_return).astype("int64") + 1
    frame = dataset.frame.copy()
    frame["neutral_target"] = labels
    gap = interval_to_timedelta(args.interval) * dataset.horizon_periods
    split = chronological_split(frame["timestamp"], gap)
    X = frame[dataset.feature_columns].to_numpy(dtype="float64")
    y = frame["neutral_target"].to_numpy(dtype="int64")

    model = XGBoostNeutralDirectionModel()
    model.fit(X[split.train], y[split.train], X[split.validation], y[split.validation])
    probabilities = model.predict_proba_classes(X[split.test])
    predicted = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predicted == y[split.test]))
    coverage = float(np.mean(predicted != 1))
    confident = predicted != 1
    confident_correct = confident & (predicted == y[split.test])
    confident_count = int(confident.sum())
    confident_accuracy = float(confident_correct.sum() / confident_count) if confident_count else None
    labels_share = {name: float(np.mean(y[split.test] == value)) for name, value in {"down": 0, "neutral": 1, "up": 2}.items()}

    print(f"Neutral direction: {args.symbol} {args.interval} {args.horizon}")
    print(f"neutral threshold: {args.neutral_return:.2%}")
    print(f"test accuracy: {accuracy:.1%}")
    print(f"non-neutral coverage: {coverage:.1%}")
    print(f"non-neutral accuracy: {'n/a' if confident_accuracy is None else f'{confident_accuracy:.1%}'}")
    print(f"test labels: {labels_share}")

    if args.no_save:
        return 0

    trained_at = datetime.now(timezone.utc)
    model_id = make_model_id("xgboost-neutral-direction", args.symbol, args.interval, args.horizon, trained_at)
    metadata = {
        "trained_at": trained_at.isoformat(),
        "provider": settings.market_data_provider,
        "symbol": args.symbol.upper(),
        "interval": args.interval,
        "horizon": args.horizon,
        "horizon_periods": dataset.horizon_periods,
        "target": "neutral_target",
        "label_mapping": {"0": "down", "1": "neutral", "2": "up"},
        "neutral_return": args.neutral_return,
        "status": "shadow",
        "feature_columns": dataset.feature_columns,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "feature_config": dataset.config.to_dict(),
        "params": model.params(),
        "training_info": model.training_info(),
        "split": split.describe(frame["timestamp"]),
        "metrics": {
            "test_accuracy": accuracy,
            "non_neutral_coverage": coverage,
            "non_neutral_accuracy": confident_accuracy,
            "non_neutral_lower_95": wilson_lower_bound(int(confident_correct.sum()), confident_count),
            "test_labels": labels_share,
        },
    }
    directory = ModelRegistry(artifacts_dir).save(model, model_id, metadata)
    print(f"saved shadow model: {model_id}")
    print(f"artifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
