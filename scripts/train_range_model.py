"""Train the range (quantile) model, compare it with the historical baseline and save it.

The model forecasts the 10th, 50th and 90th percentiles of the future return,
i.e. an 80% range for the price at the horizon.

Run from the repository root after scripts/build_features.py:

    python scripts/train_range_model.py                    # BTCUSDT 1h, horizon 24h
    python scripts/train_range_model.py --horizon 7d
    python scripts/train_range_model.py --no-save
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
from app.ml.training.train_range import train_range_models  # noqa: E402


def fmt(value: float | None, spec: str) -> str:
    return format("n/a", spec.replace(".3f", "").replace("f", "")) if value is None else format(value, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS))
    parser.add_argument("--horizon", default="24h")
    parser.add_argument("--no-save", action="store_true")
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
        print(f"No feature dataset at {dataset_path}. Run scripts/build_features.py --horizon {args.horizon}.", file=sys.stderr)
        return 1

    frame = storage.load_features(provider, symbol, args.interval, args.horizon)
    dataset_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = dataset_meta["feature_columns"]
    target = dataset_meta["target_return_column"]
    gap = interval_to_timedelta(args.interval) * dataset_meta["horizon_periods"]

    result = train_range_models(frame, features, target, gap)

    print()
    for name in ("train", "validation", "test"):
        info = result.split_info[name]
        print(f"{name:<11}: {info['rows']:>6} rows  {info['start'][:10]} .. {info['end'][:10]}")
    print()
    header = f"{'model':<22}{'split':<12}{'coverage':>10}{'width, %':>10}{'pinball':>11}{'width~|ret|':>13}"
    print(header)
    print("-" * len(header))
    for evaluation in result.evaluations:
        for split in ("validation", "test"):
            m = evaluation.metrics[split]
            print(
                f"{evaluation.name:<22}{split:<12}{m['coverage']:>9.1%} {m['mean_width'] * 100:>9.2f}"
                f"{m['pinball_mean']:>11.6f}{fmt(m['width_correlation'], '>13.3f')}"
            )

    model_eval = result.get("xgboost_quantile")
    baseline_eval = result.get("historical_quantiles")
    improvement = {
        split: 1 - model_eval.metrics[split]["pinball_mean"] / baseline_eval.metrics[split]["pinball_mean"]
        for split in ("validation", "test")
    }
    print()
    print(f"Nominal coverage            : {model_eval.metrics['test']['nominal_coverage']:.0%}")
    print(f"Pinball loss vs baseline    : validation {improvement['validation']:+.1%}, test {improvement['test']:+.1%} (positive = better)")
    print(f"Trees used                  : {model_eval.model.training_info()['best_iteration']}")

    if args.no_save:
        return 0

    trained_at = datetime.now(timezone.utc)
    model_id = make_model_id(model_eval.name.replace("_", "-"), symbol, args.interval, args.horizon, trained_at)
    metadata = {
        "trained_at": trained_at.isoformat(),
        "provider": provider,
        "symbol": symbol,
        "interval": args.interval,
        "horizon": args.horizon,
        "horizon_periods": dataset_meta["horizon_periods"],
        "target": target,
        "quantiles": list(model_eval.model.quantiles),
        "feature_columns": features,
        "feature_pipeline_version": dataset_meta["feature_pipeline_version"],
        "feature_config": dataset_meta["config"],
        "params": model_eval.model.params(),
        "training_info": model_eval.model.training_info(),
        "split": result.split_info,
        "metrics": model_eval.metrics,
        "selection_metric": "validation pinball loss (early stopping)",
        "improvement_vs_baseline": {"baseline": baseline_eval.name, "pinball": improvement},
        "baselines": {baseline_eval.name: baseline_eval.metrics},
        "dataset": {
            "path": str(dataset_path.relative_to(data_dir)),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "source_file": dataset_meta.get("source_file"),
            "source_sha256": dataset_meta.get("source_sha256"),
        },
        "code_version": git_revision(),
        "environment": environment_info(),
    }
    directory = ModelRegistry(artifacts_dir).save(model_eval.model, model_id, metadata)
    print()
    print(f"Saved model      : {model_id}")
    print(f"Artifacts        : {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
