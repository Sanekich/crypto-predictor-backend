"""Walk-forward evaluation helpers for time-series direction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.ml.evaluation.metrics import direction_metrics
from app.ml.models.direction_model import XGBoostDirectionModel


@dataclass(frozen=True)
class WalkForwardFold:
    train: np.ndarray
    early_stopping: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


def expanding_folds(
    timestamps: pd.Series,
    horizon_periods: int,
    folds: int = 4,
    min_train_fraction: float = 0.5,
) -> list[WalkForwardFold]:
    """Create expanding train folds with purge gaps and validation subdivisions."""
    n = len(timestamps)
    if folds < 1 or n < 100:
        raise ValueError("not enough data for walk-forward evaluation")
    test_size = max(1, int(n * (1 - min_train_fraction) / (folds + 1)))
    first_test = int(n * min_train_fraction)
    result: list[WalkForwardFold] = []

    for fold in range(folds):
        test_start = first_test + fold * test_size
        test_end = min(n, test_start + test_size)
        validation_end = test_start - horizon_periods
        validation_start = max(0, validation_end - test_size)
        train_end = max(0, validation_start - horizon_periods)
        validation = np.arange(validation_start, validation_end)
        if train_end <= horizon_periods or len(validation) < horizon_periods * 2 or test_end <= test_start:
            continue
        half = len(validation) // 2
        early_end = validation_start + half
        calibration_start = early_end + horizon_periods
        if calibration_start >= validation_end:
            continue
        result.append(
            WalkForwardFold(
                train=np.arange(0, train_end),
                early_stopping=np.arange(validation_start, early_end),
                calibration=np.arange(calibration_start, test_start),
                test=np.arange(test_start, test_end),
            )
        )
    return result


def evaluate_direction_walk_forward(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon_periods: int,
    folds: int = 4,
) -> list[dict[str, Any]]:
    """Train a fresh model on each expanding fold and return test metrics."""
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    X = frame[feature_columns].to_numpy(dtype="float64")
    y = frame[target_column].to_numpy(dtype="float64")
    results: list[dict[str, Any]] = []

    for number, split in enumerate(expanding_folds(frame["timestamp"], horizon_periods, folds), start=1):
        model = XGBoostDirectionModel()
        model.fit(X[split.train], y[split.train], X[split.early_stopping], y[split.early_stopping])
        model.calibrate(X[split.calibration], y[split.calibration])
        probabilities = model.predict_proba_up(X[split.test])
        metrics = direction_metrics(y[split.test], probabilities, horizon_periods)
        metrics["fold"] = number
        metrics["test_start"] = frame["timestamp"].iloc[split.test[0]].isoformat()
        metrics["test_end"] = frame["timestamp"].iloc[split.test[-1]].isoformat()
        results.append(metrics)
    return results
