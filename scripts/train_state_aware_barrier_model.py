"""Train an isolated state-aware barrier model; existing models are never changed."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.versioning import environment_info, git_revision  # noqa: E402
from app.ml.barrier.configs import BARRIER_CONFIGS, FEATURE_SET_LABELS  # noqa: E402
from app.ml.barrier.dataset import build_barrier_dataset, load_market_data  # noqa: E402
from app.ml.barrier.training import train_barrier_model  # noqa: E402
from app.ml.experiments.state_features import build_state_features  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry  # noqa: E402

MODEL_TYPE = "xgboost_state_aware_barrier"
ASCII_TARGET_LABELS = {"move_1pct_24h": "+/-1% within 24h"}
ASCII_FEATURE_LABELS = {"candles": "candles", "all_data": "candles + derivatives"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--config", default="move_1pct_24h", choices=("move_1pct_24h",))
    parser.add_argument("--feature-set", default="candles", choices=("candles", "all_data"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--shadow-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_save and not args.shadow_only:
        print("Refusing to save a state-aware model without --shadow-only", file=sys.stderr)
        return 2

    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    artifacts_dir = settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    try:
        ohlcv, derivatives = load_market_data(data_dir, settings.market_data_provider, args.symbol.upper(), args.interval, args.feature_set)
        base = build_barrier_dataset(ohlcv, args.interval, BARRIER_CONFIGS[args.config], args.feature_set, derivatives)
        state = build_state_features(base.frame)
        state_columns = list(state.columns)
        augmented = pd.concat([base.frame.reset_index(drop=True), state.reset_index(drop=True)], axis=1)
        augmented = augmented.dropna(subset=state_columns).reset_index(drop=True)
        dataset = replace(base, frame=augmented, feature_columns=[*base.feature_columns, *state_columns], rows_before_filter=len(base.frame))
        result = train_barrier_model(dataset, folds=args.folds)
    except Exception as exc:
        print(f"Cannot prepare state-aware barrier dataset: {exc}", file=sys.stderr)
        return 1

    test = result.metrics["test"]
    wf = result.walk_forward
    pooled = wf.get("pooled_signals", {}).get("0.75", {})
    print("State-aware barrier: causal regime/shock features, status shadow")
    print(f"Target              : {ASCII_TARGET_LABELS[args.config]}")
    print(f"Feature set         : {ASCII_FEATURE_LABELS[args.feature_set]} + state/shock")
    print(f"Rows                : {len(dataset.frame)} ({len(dataset.feature_columns)} features)")
    print(f"Test accuracy       : {test['accuracy']:.1%} (majority {test['majority_baseline_accuracy']:.1%})" if test["accuracy"] is not None else "Test accuracy       : n/a")
    print(f"Walk-forward skill  : {wf['brier_skill_mean']:+.4f} ({wf['positive_folds']}/{wf['folds']} positive folds)" if wf["brier_skill_mean"] is not None else "Walk-forward skill  : n/a")
    print(f"Signals >= 75%%      : {pooled.get('independent_resolved', 0)} resolved, hit {pooled.get('independent_hit_rate') if pooled.get('independent_hit_rate') is not None else 'n/a'}")
    print("Status              : SHADOW")

    if not args.no_save:
        trained_at = datetime.now(timezone.utc)
        model_id = f"state-aware-barrier-{args.config.replace('_', '-')}-{args.feature_set.replace('_', '-')}-{args.symbol.lower()}-{args.interval}-{trained_at.strftime('%Y%m%d-%H%M%S')}"
        registry = ModelRegistry(artifacts_dir / "experiments")
        metadata = {
            "trained_at": trained_at.isoformat(),
            "provider": settings.market_data_provider,
            "symbol": args.symbol.upper(),
            "interval": args.interval,
            "logic": "state_aware_barrier",
            "barrier": BARRIER_CONFIGS[args.config].describe(),
            "feature_set": args.feature_set,
            "feature_set_label": f"{FEATURE_SET_LABELS[args.feature_set]} + state/shock",
            "variant": f"{args.config}:state_aware:{args.feature_set}",
            "status": "shadow",
            "feature_columns": dataset.feature_columns,
            "feature_config": dataset.feature_config.to_dict(),
            "dataset": dataset.describe(),
            "metrics": result.metrics,
            "walk_forward": result.walk_forward,
            "split": result.split_info,
            "gate": result.gate,
            "training": result.model.training_info(),
            "params": result.model.params(),
            "code_version": git_revision(),
            "environment": environment_info(),
        }
        registry.save(result.model, model_id, metadata, model_type=MODEL_TYPE)
        print(f"Saved model        : {model_id}")
    else:
        print("Saved model        : no-save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
