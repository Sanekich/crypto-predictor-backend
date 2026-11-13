"""Train the new-logic (barrier) direction models: "which comes first, +X% or -X%?".

Run from the repository root:

    python scripts/train_barrier_models.py                               # every configuration x both feature sets
    python scripts/train_barrier_models.py --config move_1pct_24h
    python scripts/train_barrier_models.py --feature-set candles --no-save

Models are saved to artifacts/barrier/models/<model_id>/, apart from the old
logic in artifacts/models. Status "approved" means the target was reached on
pooled walk-forward signals (see app/ml/barrier/training.py); otherwise "shadow".
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
from app.data.intervals import INTERVAL_MS  # noqa: E402
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION  # noqa: E402
from app.ml.barrier.configs import (  # noqa: E402
    ARTIFACTS_SUBDIR,
    BARRIER_CONFIGS,
    FEATURE_SET_LABELS,
    FEATURE_SETS,
    MODEL_TYPE,
    SIGNAL_THRESHOLD,
    variant_key,
)
from app.ml.barrier.dataset import build_barrier_dataset, load_market_data  # noqa: E402
from app.ml.barrier.training import train_barrier_model  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry  # noqa: E402


def pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS))
    parser.add_argument("--config", default="all", choices=["all", *BARRIER_CONFIGS])
    parser.add_argument("--feature-set", default="all", choices=["all", *FEATURE_SETS])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    provider, symbol = settings.market_data_provider, args.symbol.upper()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    artifacts_dir = settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    registry = ModelRegistry(artifacts_dir / ARTIFACTS_SUBDIR)

    configs = list(BARRIER_CONFIGS.values()) if args.config == "all" else [BARRIER_CONFIGS[args.config]]
    feature_sets = list(FEATURE_SETS) if args.feature_set == "all" else [args.feature_set]

    data_cache: dict[str, tuple] = {}
    summary: list[str] = []
    for feature_set in feature_sets:
        try:
            data_cache[feature_set] = load_market_data(data_dir, provider, symbol, args.interval, feature_set)
        except FileNotFoundError as exc:
            print(f"[{feature_set}] skipped: {exc}", file=sys.stderr)

    for config in configs:
        for feature_set in feature_sets:
            if feature_set not in data_cache:
                continue
            ohlcv, derivatives = data_cache[feature_set]
            dataset = build_barrier_dataset(ohlcv, args.interval, config, feature_set, derivatives)
            info = dataset.describe()
            # ASCII only: the output is shown in Windows consoles and in the operations log
            print(f"\n===== {config.key} (+-{config.threshold:.1%} within {config.max_horizon})  |  features: {feature_set}")
            print(f"Rows                 : {info['rows']:>6}  {info['first_timestamp'][:10]} .. {info['last_timestamp'][:10]}")
            shares = info["outcome_shares"]
            print(f"Outcomes             : up {shares['up']:.1%}  down {shares['down']:.1%}  none {shares['none']:.1%}  both {shares['both']:.1%}")

            result = train_barrier_model(dataset, folds=args.folds)
            test = result.metrics["test"]
            wf = result.walk_forward
            pooled = wf["pooled_signals"].get(f"{SIGNAL_THRESHOLD:.2f}", {})
            print(f"Test accuracy        : {pct(test['accuracy'])}  (majority {pct(test['majority_baseline_accuracy'])}, resolved {test['resolved_share']:.1%})")
            print(f"Test max confidence  : {pct(test['max_confidence'])}")
            folds = ", ".join(f"{s:+.3f}" for s in wf["brier_skill_by_fold"])
            mean = wf["brier_skill_mean"]
            print(f"Walk-forward skill   : {'n/a' if mean is None else f'{mean:+.4f}'}  [{folds}]  positive {wf['positive_folds']}/{wf['folds']}")
            print(
                f"Signals >= {SIGNAL_THRESHOLD:.0%}       : {pooled.get('independent_signals', 0)} independent, "
                f"{pooled.get('independent_resolved', 0)} resolved, hit {pct(pooled.get('independent_hit_rate'))}, "
                f"95% lower bound {pct(pooled.get('hit_rate_lower_95'))}"
            )
            print(f"Status               : {result.status.upper()}")
            for reason in result.gate["reasons"]:
                print(f"  - {reason}")
            summary.append(
                f"{config.key:<16} {feature_set:<9} skill {'  n/a ' if mean is None else f'{mean:+.4f}'} "
                f"{wf['positive_folds']}/{wf['folds']}  signals {pooled.get('independent_resolved', 0):>3} "
                f"hit {pct(pooled.get('independent_hit_rate'))}  {result.status}"
            )

            if args.no_save:
                continue
            trained_at = datetime.now(timezone.utc)
            stamp = trained_at.strftime("%Y%m%d-%H%M%S")
            model_id = f"barrier-{config.key.replace('_', '-')}-{feature_set.replace('_', '-')}-{symbol.lower()}-{args.interval}-{stamp}"
            metadata = {
                "trained_at": trained_at.isoformat(),
                "provider": provider,
                "symbol": symbol,
                "interval": args.interval,
                "logic": "barrier",
                "horizon": config.max_horizon,
                "barrier": config.describe(),
                "max_periods": dataset.max_periods,
                "feature_set": feature_set,
                "feature_set_label": FEATURE_SET_LABELS[feature_set],
                "variant": variant_key(config.key, feature_set),
                "status": result.status,
                "signal_threshold": SIGNAL_THRESHOLD,
                "gate": result.gate,
                "walk_forward": wf,
                "feature_columns": dataset.feature_columns,
                "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
                "feature_config": dataset.feature_config.to_dict(),
                "dataset": {k: v for k, v in info.items() if k not in ("feature_columns", "feature_config")},
                "split": result.split_info,
                "metrics": result.metrics,
                "training": result.model.training_info(),
                "params": result.model.params(),
                "code_version": git_revision(),
                "environment": environment_info(),
            }
            directory = registry.save(result.model, model_id, metadata, model_type=MODEL_TYPE)
            print(f"Saved model          : {model_id}")
            print(f"Artifacts            : {directory}")

    if not summary:
        print("Nothing was trained: data is missing.", file=sys.stderr)
        return 1
    print("\n===== Summary (walk-forward, pooled signals >= 75%)")
    for line in summary:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
