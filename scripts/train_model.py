"""Train baselines and XGBoost on the feature dataset, compare them and save the model.

Run from the repository root after scripts/build_features.py:

    python scripts/train_model.py                      # BTCUSDT 1h, horizon 24h
    python scripts/train_model.py --horizon 4h
    python scripts/train_model.py --no-save            # experiment without saving
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.versioning import environment_info, git_revision  # noqa: E402
from app.data.intervals import INTERVAL_MS, interval_to_timedelta  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry, make_model_id  # noqa: E402
from app.ml.training.train import train_and_evaluate  # noqa: E402


def fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def print_table(result) -> None:
    header = f"{'model':<16}{'split':<12}{'MAE, bps':>10}{'RMSE':>10}{'direction':>11}{'corr':>8}"
    print(header)
    print("-" * len(header))
    for evaluation in result.evaluations:
        for split in ("validation", "test"):
            m = evaluation.metrics[split]
            print(
                f"{evaluation.name:<16}{split:<12}{m['mae_bps']:>10.1f}{m['rmse']:>10.5f}"
                f"{m['direction_accuracy']:>10.1%} {fmt(m['correlation'], '>7.3f')}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS))
    parser.add_argument("--horizon", default="24h")
    parser.add_argument("--no-save", action="store_true", help="do not save the model")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    provider = settings.market_data_provider
    symbol = args.symbol.upper()

    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    artifacts_dir = (
        settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    )
    storage = MarketDataStorage(data_dir)
    dataset_path = storage.features_path(provider, symbol, args.interval, args.horizon)
    meta_path = storage.features_meta_path(provider, symbol, args.interval, args.horizon)
    if not dataset_path.exists() or not meta_path.exists():
        print(f"No feature dataset at {dataset_path}. Run scripts/build_features.py first.", file=sys.stderr)
        return 1

    frame = storage.load_features(provider, symbol, args.interval, args.horizon)
    dataset_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = dataset_meta["feature_columns"]
    target = dataset_meta["target_return_column"]
    gap = interval_to_timedelta(args.interval) * dataset_meta["horizon_periods"]

    result = train_and_evaluate(frame, features, target, gap)

    print()
    for name, info in result.split_info.items():
        if name != "gap":
            print(f"{name:<11}: {info['rows']:>6} rows  {info['start'][:10]} .. {info['end'][:10]}")
    print(f"gap        : {result.split_info['gap']} removed before validation and test (purging)")
    print()
    print_table(result)

    xgb = result.get("xgboost")
    best_baseline = result.best_baseline()
    xgb_val, base_val = xgb.metrics["validation"]["rmse"], best_baseline.metrics["validation"]["rmse"]
    improvement = (base_val - xgb_val) / base_val
    print()
    print(f"Best baseline on validation: {best_baseline.name} (RMSE {base_val:.5f})")
    print(f"XGBoost vs best baseline   : {improvement:+.2%} RMSE on validation (positive = better)")
    print(f"XGBoost trees used         : {xgb.model.training_info()['best_iteration']}")

    importance = xgb.model.feature_importance(features)
    print("Top features (gain)        : " + ", ".join(f"{k} {v:.0%}" for k, v in list(importance.items())[:5]))

    if args.no_save:
        return 0

    trained_at = datetime.now(timezone.utc)
    model_id = make_model_id(xgb.name, symbol, args.interval, args.horizon, trained_at)
    metadata = {
        "trained_at": trained_at.isoformat(),
        "provider": provider,
        "symbol": symbol,
        "interval": args.interval,
        "horizon": args.horizon,
        "horizon_periods": dataset_meta["horizon_periods"],
        "target": target,
        "feature_columns": features,
        "feature_pipeline_version": dataset_meta["feature_pipeline_version"],
        "feature_config": dataset_meta["config"],
        "params": xgb.model.params(),
        "training_info": xgb.model.training_info(),
        "split": result.split_info,
        "metrics": xgb.metrics,
        "selection_metric": "validation rmse",
        "improvement_vs_best_baseline": {"baseline": best_baseline.name, "validation_rmse": improvement},
        "baselines": {e.name: e.metrics for e in result.evaluations if e.name != xgb.name},
        "feature_importance": importance,
        "dataset": {
            "path": str(dataset_path.relative_to(data_dir)),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "source_file": dataset_meta.get("source_file"),
            "source_sha256": dataset_meta.get("source_sha256"),
        },
        "code_version": git_revision(),
        "environment": environment_info(),
    }
    directory = ModelRegistry(artifacts_dir).save(xgb.model, model_id, metadata)
    print()
    print(f"Saved model      : {model_id}")
    print(f"Artifacts        : {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
