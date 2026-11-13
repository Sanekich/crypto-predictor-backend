"""Train a calibrated direction model for a short horizon and verify its high-confidence signals.

Run from the repository root:

    python scripts/train_direction_model.py --horizon 1h
    python scripts/train_direction_model.py --horizon 4h
    python scripts/train_direction_model.py --horizon 4h --no-save

The model is saved with status "approved" only if its >=75% signals passed the
test-period checks (docs/direction_prediction_plan.md, section 7). Otherwise it
is saved as "shadow": the site shows its signals as unverified.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.versioning import environment_info, git_revision  # noqa: E402
from app.data.intervals import INTERVAL_MS, interval_to_timedelta  # noqa: E402
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION  # noqa: E402
from app.features.feature_sets import FEATURE_SET_LABELS, load_direction_dataset  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry, make_model_id  # noqa: E402
from app.ml.training.train_direction import SIGNAL_THRESHOLD, train_direction_model, walk_forward_skill  # noqa: E402


def pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS))
    parser.add_argument("--horizon", default="4h")
    parser.add_argument("--feature-set", choices=("v1", "v2_funding_basis", "v3_open_interest"), default="v1")
    parser.add_argument("--shadow-only", action="store_true", help="save the artifact only as shadow, never approved")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    provider, symbol = settings.market_data_provider, args.symbol.upper()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    artifacts_dir = (
        settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    )

    try:
        dataset = load_direction_dataset(data_dir, provider, symbol, args.interval, args.horizon, args.feature_set)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Cannot prepare {args.feature_set} dataset: {exc}", file=sys.stderr)
        return 1
    gap = interval_to_timedelta(args.interval) * dataset.horizon_periods
    result = train_direction_model(
        dataset.frame, dataset.feature_columns, dataset.target_direction_column, dataset.horizon_periods, gap
    )

    print("Walk-forward (5 folds)...", flush=True)
    skill = walk_forward_skill(
        dataset.frame, dataset.feature_columns, dataset.target_direction_column, dataset.horizon_periods
    )

    test = result.metrics["test"]
    print()
    for name in ("train", "early_stopping", "calibration", "test"):
        info = result.split_info[name]
        print(f"{name:<15}: {info['rows']:>6} rows  {info['start'][:10]} .. {info['end'][:10]}")
    print()
    print(f"Feature set          : {args.feature_set} ({FEATURE_SET_LABELS[args.feature_set]})")
    print(f"Test accuracy        : {test['accuracy']:.1%}")
    print(f"Always UP baseline   : {test['always_up_accuracy']:.1%}")
    print(f"Always DOWN baseline : {test['always_down_accuracy']:.1%}")
    print(f"Majority baseline    : {test['majority_baseline_accuracy']:.1%}")
    print(f"Brier score          : {test['brier']:.4f}  (base rate: {test['brier_base_rate']:.4f}, lower is better)")
    print(f"Max confidence (test): {test['max_confidence']:.1%}")
    print()
    print(f"{'confidence':>10} {'signals':>8} {'time':>7} {'hit rate':>9} {'independent':>12} {'hit':>7} {'lower 95%':>10}")
    for threshold, s in test["signals"].items():
        print(
            f"{'>= ' + threshold:>10} {s['signals']:>8} {pct(s['share_of_time'])} {pct(s['hit_rate']):>9}"
            f" {s['independent_signals']:>12} {pct(s['independent_hit_rate'])} {pct(s['hit_rate_lower_95']):>10}"
        )
    print()
    folds = ", ".join(f"{s:+.3f}" for s in skill["brier_skill_by_fold"])
    print(f"Walk-forward skill   : {skill['brier_skill_mean']:+.4f}  [{folds}]  positive {skill['positive_folds']}/{skill['folds']}")
    print(f"Direction shown      : {'YES (beats base rate on most folds)' if skill['has_skill'] else 'NO (no stable skill)'}")
    saved_status = "shadow" if args.shadow_only else result.status
    print(f"Status: {saved_status.upper()}")
    for reason in result.gate["reasons"]:
        print(f"  - {reason}")

    if args.no_save:
        return 0

    trained_at = datetime.now(timezone.utc)
    model_prefixes = {
        "v1": "xgboost-direction",
        "v2_funding_basis": "xgboost-direction-v2-funding-basis",
        "v3_open_interest": "xgboost-direction-v3-open-interest",
    }
    model_prefix = model_prefixes[args.feature_set]
    model_id = make_model_id(model_prefix, symbol, args.interval, args.horizon, trained_at)
    metadata = {
        "trained_at": trained_at.isoformat(),
        "provider": provider,
        "symbol": symbol,
        "interval": args.interval,
        "horizon": args.horizon,
        "horizon_periods": dataset.horizon_periods,
        "target": dataset.target_direction_column,
        "status": saved_status,
        "feature_set": args.feature_set,
        "signal_threshold": SIGNAL_THRESHOLD,
        "gate": result.gate,
        "walk_forward": skill,
        "feature_columns": dataset.feature_columns,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "feature_config": dataset.config.to_dict(),
        "params": result.model.params(),
        "training_info": result.model.training_info(),
        "split": result.split_info,
        "metrics": result.metrics,
        "code_version": git_revision(),
        "environment": environment_info(),
    }
    registry_model_type = {
        "v1": "xgboost_direction",
        "v2_funding_basis": "xgboost_direction_v2",
        "v3_open_interest": "xgboost_direction_v3",
    }[args.feature_set]
    directory = ModelRegistry(artifacts_dir).save(result.model, model_id, metadata, model_type=registry_model_type)
    print()
    print(f"Saved model : {model_id}")
    print(f"Artifacts   : {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
