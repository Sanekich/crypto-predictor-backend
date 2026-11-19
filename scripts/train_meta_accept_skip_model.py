"""Train an isolated ACCEPT/SKIP meta-model for the +/-1% / 24h barrier model."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.versioning import environment_info, git_revision  # noqa: E402
from app.data.intervals import interval_to_timedelta  # noqa: E402
from app.ml.barrier.configs import BARRIER_CONFIGS, FEATURE_SET_LABELS  # noqa: E402
from app.ml.barrier.dataset import build_barrier_dataset, load_market_data  # noqa: E402
from app.ml.evaluation.splits import chronological_split  # noqa: E402
from app.ml.evaluation.walk_forward import expanding_folds  # noqa: E402
from app.ml.models.direction_model import XGBoostDirectionModel  # noqa: E402
from app.ml.models.meta_accept_skip_model import MetaAcceptSkipModel  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry  # noqa: E402

MODEL_TYPE = "xgboost_meta_accept_skip"
CONFIG_KEY = "move_1pct_24h"
META_FEATURES = ["primary_probability", "primary_confidence", "confidence_squared"]
ASCII_FEATURE_LABELS = {"candles": "candles", "all_data": "candles + derivatives"}


def _labeled(indices: np.ndarray, label: np.ndarray) -> np.ndarray:
    return indices[~np.isnan(label[indices])]


def _meta_features(probability: np.ndarray) -> np.ndarray:
    confidence = np.maximum(probability, 1.0 - probability)
    return np.column_stack([probability, confidence, confidence * confidence])


def _fit_primary(X: np.ndarray, label: np.ndarray, train: np.ndarray, early: np.ndarray, calibration: np.ndarray) -> XGBoostDirectionModel:
    model = XGBoostDirectionModel()
    model.fit(X[train], label[train], X[early], label[early])
    model.calibrate(X[calibration], label[calibration])
    return model


def _correct_primary(outcome: np.ndarray, probability: np.ndarray) -> np.ndarray:
    direction_up = probability >= 0.5
    return np.where(direction_up, outcome == "up", outcome == "down").astype(int)


def _evaluate(primary_probability: np.ndarray, outcome: np.ndarray, meta_probability: np.ndarray) -> dict[str, float | int | None]:
    resolved = np.isin(outcome, ["up", "down"])
    if not resolved.any():
        return {"resolved_rows": 0, "accepted": 0, "coverage": None, "primary_accuracy": None, "accepted_accuracy": None, "skip_rate": None, "accept_precision": None}
    primary_correct = _correct_primary(outcome, primary_probability)
    accepted = (meta_probability >= 0.5) & resolved
    accepted_count = int(accepted.sum())
    return {
        "resolved_rows": int(resolved.sum()),
        "accepted": accepted_count,
        "coverage": float(accepted_count / resolved.sum()),
        "primary_accuracy": float(primary_correct[resolved].mean()),
        "accepted_accuracy": float(primary_correct[accepted].mean()) if accepted_count else None,
        "skip_rate": float(1.0 - accepted_count / resolved.sum()),
        "accept_precision": float(primary_correct[accepted].mean()) if accepted_count else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--feature-set", default="candles", choices=("candles", "all_data"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--shadow-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_save and not args.shadow_only:
        print("Refusing to save a meta-model without --shadow-only", file=sys.stderr)
        return 2

    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    artifacts_dir = settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    config = BARRIER_CONFIGS[CONFIG_KEY]
    ohlcv, derivatives = load_market_data(data_dir, settings.market_data_provider, args.symbol.upper(), args.interval, args.feature_set)
    dataset = build_barrier_dataset(ohlcv, args.interval, config, args.feature_set, derivatives)
    frame = dataset.frame.sort_values("timestamp").reset_index(drop=True)
    X = frame[dataset.feature_columns].to_numpy(dtype="float64")
    label = frame["label"].to_numpy(dtype="float64")
    outcome = frame["outcome"].to_numpy(dtype=object)
    ts = frame["timestamp"]
    gap = interval_to_timedelta(dataset.interval) * dataset.max_periods
    split = chronological_split(ts, gap, 0.2, 0.15)
    half = len(split.validation) // 2
    early_all = split.validation[:half]
    calibration_all = split.validation[half + dataset.max_periods:]
    train, early, calibration = (_labeled(index, label) for index in (split.train, early_all, calibration_all))
    primary = _fit_primary(X, label, train, early, calibration)
    calibration_probability = primary.raw_proba_up(X[calibration_all])
    calibration_resolved = np.isin(outcome[calibration_all], ["up", "down"])
    meta = MetaAcceptSkipModel(META_FEATURES)
    meta.fit(_meta_features(calibration_probability[calibration_resolved]), _correct_primary(outcome[calibration_all][calibration_resolved], calibration_probability[calibration_resolved]))
    test_probability = primary.raw_proba_up(X[split.test])
    test_meta_probability = meta.predict_accept_proba(_meta_features(test_probability))
    test_metrics = _evaluate(test_probability, outcome[split.test], test_meta_probability)

    folds = []
    for number, fold in enumerate(expanding_folds(ts, dataset.max_periods, args.folds), start=1):
        fold_train, fold_early, fold_calibration = (_labeled(index, label) for index in (fold.train, fold.early_stopping, fold.calibration))
        if min(len(fold_train), len(fold_early), len(fold_calibration)) < 200:
            continue
        fold_primary = _fit_primary(X, label, fold_train, fold_early, fold_calibration)
        cal_prob = fold_primary.raw_proba_up(X[fold.calibration])
        cal_resolved = np.isin(outcome[fold.calibration], ["up", "down"])
        fold_meta = MetaAcceptSkipModel(META_FEATURES)
        fold_meta.fit(_meta_features(cal_prob[cal_resolved]), _correct_primary(outcome[fold.calibration][cal_resolved], cal_prob[cal_resolved]))
        fold_prob = fold_primary.raw_proba_up(X[fold.test])
        fold_meta_prob = fold_meta.predict_accept_proba(_meta_features(fold_prob))
        metrics = _evaluate(fold_prob, outcome[fold.test], fold_meta_prob)
        metrics.update(fold=number, test_start=ts.iloc[fold.test[0]].isoformat(), test_end=ts.iloc[fold.test[-1]].isoformat())
        folds.append(metrics)

    accepted_accuracies = [row["accepted_accuracy"] for row in folds if row["accepted_accuracy"] is not None]
    positive = sum(1 for row in folds if row["accepted_accuracy"] is not None and row["accepted_accuracy"] > row["primary_accuracy"])
    print("Meta-model ACCEPT/SKIP: shadow")
    print("Primary target       : +/-1% within 24h")
    print(f"Feature set          : {ASCII_FEATURE_LABELS[args.feature_set]} + primary confidence")
    print(f"Test primary accuracy: {test_metrics['primary_accuracy']:.1%}" if test_metrics["primary_accuracy"] is not None else "Test primary accuracy: n/a")
    print(f"Test accepted accuracy: {test_metrics['accepted_accuracy']:.1%}" if test_metrics["accepted_accuracy"] is not None else "Test accepted accuracy: n/a")
    print(f"Accepted coverage    : {test_metrics['coverage']:.1%}" if test_metrics["coverage"] is not None else "Accepted coverage    : n/a")
    print(f"Skipped signals      : {test_metrics['skip_rate']:.1%}" if test_metrics["skip_rate"] is not None else "Skipped signals      : n/a")
    print(f"Walk-forward gain    : {positive}/{len(folds)} folds improved")
    print("Status               : SHADOW")

    if not args.no_save:
        trained_at = datetime.now(timezone.utc)
        model_id = f"meta-accept-skip-{CONFIG_KEY.replace('_', '-')}-{args.feature_set.replace('_', '-')}-{args.symbol.lower()}-{args.interval}-{trained_at.strftime('%Y%m%d-%H%M%S')}"
        metadata = {
            "trained_at": trained_at.isoformat(), "provider": settings.market_data_provider, "symbol": args.symbol.upper(), "interval": args.interval,
            "logic": "meta_accept_skip", "horizon": config.max_horizon, "barrier": config.describe(), "feature_set": args.feature_set,
            "feature_set_label": f"{FEATURE_SET_LABELS[args.feature_set]} + primary confidence", "variant": f"{CONFIG_KEY}:meta_accept_skip:{args.feature_set}", "status": "shadow",
            "feature_columns": META_FEATURES, "dataset": dataset.describe(), "metrics": {"test": test_metrics},
            "walk_forward": {"folds": len(folds), "by_fold": folds, "positive_folds": positive, "accepted_accuracy_mean": float(np.mean(accepted_accuracies)) if accepted_accuracies else None},
            "training": meta.training_info(), "params": meta.params(), "primary_model": "xgboost_barrier_direction", "code_version": git_revision(), "environment": environment_info(),
        }
        registry = ModelRegistry(artifacts_dir / "experiments")
        registry.save(meta, model_id, metadata, model_type=MODEL_TYPE)
        print(f"Saved model         : {model_id}")
    else:
        print("Saved model         : no-save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
