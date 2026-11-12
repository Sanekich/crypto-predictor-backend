"""Walk-forward evaluation helpers for time-series direction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.ml.evaluation.metrics import direction_metrics, independent_sample_indices, interval_metrics
from app.ml.models.direction_model import XGBoostDirectionModel
from app.ml.models.quantile_model import HistoricalQuantileBaseline, XGBoostQuantileModel


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
                # ends one horizon before the test: calibration targets must not overlap test prices
                calibration=np.arange(calibration_start, validation_end),
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
        # Same procedure as production training (train_direction.py)
        model.calibrate(X[split.calibration], y[split.calibration])
        probabilities = model.predict_proba_up(X[split.test])
        metrics = direction_metrics(y[split.test], probabilities, horizon_periods)
        independent_test = independent_sample_indices(split.test, horizon_periods)
        independent_metrics = direction_metrics(y[independent_test], model.predict_proba_up(X[independent_test]), 1)
        metrics["independent_n"] = int(len(independent_test))
        metrics["independent_accuracy"] = independent_metrics["accuracy"]
        metrics["independent_brier"] = independent_metrics["brier"]
        metrics["fold"] = number
        metrics["test_start"] = frame["timestamp"].iloc[split.test[0]].isoformat()
        metrics["test_end"] = frame["timestamp"].iloc[split.test[-1]].isoformat()
        results.append(metrics)
    return results


def evaluate_range_walk_forward(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon_periods: int,
    folds: int = 4,
) -> list[dict[str, Any]]:
    """Range (quantile) model vs a constant historical range on each expanding fold."""
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    X = frame[feature_columns].to_numpy(dtype="float64")
    y = frame[target_column].to_numpy(dtype="float64")
    results: list[dict[str, Any]] = []

    for number, split in enumerate(expanding_folds(frame["timestamp"], horizon_periods, folds), start=1):
        validation = np.concatenate([split.early_stopping, split.calibration])
        model = XGBoostQuantileModel().fit(X[split.train], y[split.train], X[validation], y[validation])
        baseline = HistoricalQuantileBaseline().fit(X[split.train], y[split.train])
        metrics = interval_metrics(y[split.test], model.predict_quantiles(X[split.test]), model.quantiles)
        base = interval_metrics(y[split.test], baseline.predict_quantiles(X[split.test]), baseline.quantiles)
        metrics["pinball_skill_vs_constant"] = 1 - metrics["pinball_mean"] / base["pinball_mean"]
        metrics["fold"] = number
        metrics["test_start"] = frame["timestamp"].iloc[split.test[0]].isoformat()
        metrics["test_end"] = frame["timestamp"].iloc[split.test[-1]].isoformat()
        results.append(metrics)
    return results
