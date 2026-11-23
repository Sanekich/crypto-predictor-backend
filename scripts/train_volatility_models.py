"""Train the volatility models: "will BTC move by at least X% (either way) within N hours?".

Run from the repository root:

    python scripts/train_volatility_models.py                          # 3 configurations x 2 feature sets
    python scripts/train_volatility_models.py --config move_1pct_4h --feature-set candles
    python scripts/train_volatility_models.py --no-save

Models are saved to artifacts/volatility/models/<model_id>/. Each run prints the
walk-forward skill of the model and of a naive baseline (24h realized volatility).
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
from app.ml.barrier.dataset import load_market_data  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry  # noqa: E402
from app.ml.volatility.configs import (  # noqa: E402
    ARTIFACTS_SUBDIR,
    FEATURE_SET_LABELS,
    FEATURE_SETS,
    MODEL_TYPE,
    SIGNAL_THRESHOLD,
    VOLATILITY_CONFIGS,
    variant_key,
)
from app.ml.volatility.dataset import build_volatility_dataset  # noqa: E402
from app.ml.volatility.training import train_volatility_model  # noqa: E402


def pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def signed(value: float | None) -> str:
    return "  n/a  " if value is None else f"{value:+.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS))
    parser.add_argument("--config", default="all", choices=["all", *VOLATILITY_CONFIGS])
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

    configs = list(VOLATILITY_CONFIGS.values()) if args.config == "all" else [VOLATILITY_CONFIGS[args.config]]
    feature_sets = list(FEATURE_SETS) if args.feature_set == "all" else [args.feature_set]
    key = f"{SIGNAL_THRESHOLD:.2f}"
    summary: list[str] = []

    for feature_set in feature_sets:
        try:
            ohlcv, derivatives = load_market_data(data_dir, provider, symbol, args.interval, feature_set)
        except FileNotFoundError as exc:
            print(f"[{feature_set}] skipped: {exc}", file=sys.stderr)
            continue
        for config in configs:
            dataset = build_volatility_dataset(ohlcv, args.interval, config, feature_set, derivatives)
            info = dataset.describe()
            print(f"\n===== {config.key} (move +-{config.threshold:.0%} within {config.horizon})  |  features: {feature_set}")
            print(f"Rows                 : {info['rows']:>6}  {info['first_timestamp'][:10]} .. {info['last_timestamp'][:10]}, big move in {info['move_share']:.1%}")

            result = train_volatility_model(dataset, folds=args.folds)
            wf = result.walk_forward
            test = result.metrics["test"]
            folds_text = ", ".join(f"{s:+.3f}" for s in wf["brier_skill_by_fold"])
            print(f"Walk-forward skill   : {signed(wf['brier_skill_mean'])}  [{folds_text}]  positive {wf['positive_folds']}/{wf['folds']}")
            print(f"Naive baseline skill : {signed(wf['baseline_brier_skill_mean'])}  (24h realized volatility only)")
            aucs = ", ".join("n/a" if a is None else f"{a:.3f}" for a in wf["auc_by_fold"])
            print(f"AUC by fold          : [{aucs}]")
            print(f"Test accuracy        : {pct(test['accuracy'])}  (majority {pct(test['majority_baseline_accuracy'])}), P(move) range {test['min_p_move']:.2f}..{test['max_p_move']:.2f}")
            for kind, label in (("move_signals", "Move  signals >= 75%"), ("calm_signals", "Calm  signals >= 75%")):
                s = wf["pooled"][kind].get(key, {})
                print(f"{label} : {s.get('independent_signals', 0):>4} independent, hit {pct(s.get('independent_hit_rate'))}, 95% lower {pct(s.get('hit_rate_lower_95'))}")
            print(f"Status               : {result.status.upper()}")
            for reason in wf["gate"]["reasons"]:
                print(f"  - {reason}")
            best = wf["pooled"][wf["best_signal_kind"]].get(key, {})
            summary.append(
                f"{config.key:<14} {feature_set:<9} skill {signed(wf['brier_skill_mean'])} {wf['positive_folds']}/{wf['folds']} "
                f"baseline {signed(wf['baseline_brier_skill_mean'])}  best 75% ({wf['best_signal_kind'][:4]}) "
                f"{best.get('independent_signals', 0):>4} hit {pct(best.get('independent_hit_rate'))}  {result.status}"
            )

            if args.no_save:
                continue
            trained_at = datetime.now(timezone.utc)
            model_id = f"volatility-{config.key.replace('_', '-')}-{feature_set.replace('_', '-')}-{symbol.lower()}-{args.interval}-{trained_at:%Y%m%d-%H%M%S}"
            metadata = {
                "trained_at": trained_at.isoformat(),
                "provider": provider,
                "symbol": symbol,
                "interval": args.interval,
                "logic": "volatility",
                "horizon": config.horizon,
                "volatility": config.describe(),
                "periods": dataset.periods,
                "feature_set": feature_set,
                "feature_set_label": FEATURE_SET_LABELS[feature_set],
                "variant": variant_key(config.key, feature_set),
                "status": result.status,
                "signal_threshold": SIGNAL_THRESHOLD,
                "walk_forward": wf,
                "feature_columns": dataset.feature_columns,
                "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
                "feature_config": dataset.feature_config.to_dict(),
                "dataset": info,
                "split": result.split_info,
                "metrics": result.metrics,
                "training": result.model.training_info(),
                "params": result.model.params(),
                "code_version": git_revision(),
                "environment": environment_info(),
            }
            registry.save(result.model, model_id, metadata, model_type=MODEL_TYPE)
            print(f"Saved model          : {model_id}")

    if not summary:
        print("Nothing was trained: data is missing.", file=sys.stderr)
        return 1
    print("\n===== Summary")
    for line in summary:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
