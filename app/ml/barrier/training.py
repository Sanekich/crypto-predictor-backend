"""Training and walk-forward verification of barrier direction models.

Time periods of the saved model, oldest to newest:

    | train | gap | early stopping | gap | calibration | gap | test |

Only resolved rows (an up or down barrier was hit) are used for fitting and
calibration; the test metrics use every row, so that confident signals followed
by no move at all are counted too.

The verdict does not rely on the single test period: the same procedure is
repeated on 5 expanding walk-forward folds and the confident signals of all
their test periods are pooled. The target (approval) is the same as for the
old logic: at least 77 independent signals at >= 75% confidence, hit rate
>= 75% and its 95% lower bound >= 70% - on top of a positive Brier skill on
most folds.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.data.intervals import interval_to_timedelta
from app.ml.barrier.configs import SIGNAL_THRESHOLD
from app.ml.barrier.dataset import BarrierDataset
from app.ml.barrier.metrics import barrier_metrics, pool_signals
from app.ml.evaluation.metrics import SIGNAL_THRESHOLDS
from app.ml.evaluation.splits import chronological_split
from app.ml.evaluation.walk_forward import expanding_folds
from app.ml.models.direction_model import XGBoostDirectionModel

logger = logging.getLogger(__name__)

GATE = {"min_independent_signals": 77, "min_hit_rate": 0.75, "min_hit_rate_lower_95": 0.70}
SKILL_GATE = {"folds": 5, "min_positive_folds": 4}
MIN_FIT_ROWS = 200


def _labeled(indices: np.ndarray, label: np.ndarray) -> np.ndarray:
    return indices[~np.isnan(label[indices])]


def _fit(
    X: np.ndarray,
    label: np.ndarray,
    train: np.ndarray,
    early_stop: np.ndarray,
    calibration: np.ndarray,
    params: dict[str, Any] | None,
) -> XGBoostDirectionModel:
    model = XGBoostDirectionModel(params)
    model.fit(X[train], label[train], X[early_stop], label[early_stop])
    model.calibrate(X[calibration], label[calibration])
    return model


def evaluate_gate(pooled: dict[str, Any], threshold: float = SIGNAL_THRESHOLD) -> dict[str, Any]:
    """Has the model reached the target on pooled walk-forward signals?"""
    n = pooled["independent_resolved"]
    hit = pooled["independent_hit_rate"]
    low = pooled["hit_rate_lower_95"]
    reasons = []
    if n < GATE["min_independent_signals"]:
        reasons.append(f"only {n} independent resolved signals >= {threshold:.0%} (need {GATE['min_independent_signals']})")
    if hit is not None and hit < GATE["min_hit_rate"]:
        reasons.append(f"hit rate {hit:.1%} below {GATE['min_hit_rate']:.0%}")
    if low is not None and low < GATE["min_hit_rate_lower_95"]:
        reasons.append(f"95% lower bound {low:.1%} below {GATE['min_hit_rate_lower_95']:.0%}")
    return {
        "passed": not reasons,
        "threshold": threshold,
        "criteria": GATE,
        "reasons": reasons,
        "progress": {
            "independent_signals": min(1.0, n / GATE["min_independent_signals"]),
            "hit_rate": None if hit is None else min(1.0, hit / GATE["min_hit_rate"]),
            "hit_rate_lower_95": None if low is None else min(1.0, low / GATE["min_hit_rate_lower_95"]),
        },
    }


def walk_forward(dataset: BarrierDataset, folds: int = SKILL_GATE["folds"], params: dict[str, Any] | None = None) -> dict[str, Any]:
    frame = dataset.frame.sort_values("timestamp").reset_index(drop=True)
    X = frame[dataset.feature_columns].to_numpy(dtype="float64")
    label = frame["label"].to_numpy(dtype="float64")
    outcome = frame["outcome"].to_numpy(dtype=object)
    durations = frame["resolve_periods"].to_numpy(dtype=int)
    ts = frame["timestamp"]

    results: list[dict[str, Any]] = []
    for number, split in enumerate(expanding_folds(ts, dataset.max_periods, folds), start=1):
        train, early_stop, calibration = (_labeled(idx, label) for idx in (split.train, split.early_stopping, split.calibration))
        if min(len(train), len(early_stop), len(calibration)) < MIN_FIT_ROWS:
            continue
        model = _fit(X, label, train, early_stop, calibration, params)
        metrics = barrier_metrics(outcome[split.test], model.predict_proba_up(X[split.test]), durations[split.test])
        metrics.update(
            fold=number,
            rows=int(len(split.test)),
            test_start=ts.iloc[split.test[0]].isoformat(),
            test_end=ts.iloc[split.test[-1]].isoformat(),
        )
        results.append(metrics)

    skills = [m["brier_skill"] for m in results if m["brier_skill"] is not None]
    positive = sum(s > 0 for s in skills)
    mean_skill = float(np.mean(skills)) if skills else None
    # 4 of 5 folds by default; with fewer folds every fold must be positive
    required_positive = min(SKILL_GATE["min_positive_folds"], len(skills))
    pooled = {f"{t:.2f}": pool_signals(results, t) for t in SIGNAL_THRESHOLDS} if results else {}
    gate = evaluate_gate(pooled[f"{SIGNAL_THRESHOLD:.2f}"]) if results else evaluate_gate(pool_signals([], SIGNAL_THRESHOLD))
    return {
        "folds": len(results),
        "by_fold": results,
        "brier_skill_by_fold": skills,
        "brier_skill_mean": mean_skill,
        "positive_folds": positive,
        "accuracy_by_fold": [m["accuracy"] for m in results],
        "resolved_share": float(np.mean([m["resolved_share"] for m in results])) if results else None,
        "max_confidence": max((m["max_confidence"] for m in results), default=None),
        "required_positive_folds": required_positive,
        "has_skill": mean_skill is not None and mean_skill > 0 and positive >= required_positive,
        "criteria": SKILL_GATE,
        "pooled_signals": pooled,
        "gate": gate,
    }


@dataclass
class BarrierTrainingResult:
    model: XGBoostDirectionModel
    split_info: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    walk_forward: dict[str, Any]

    @property
    def gate(self) -> dict[str, Any]:
        return self.walk_forward["gate"]

    @property
    def status(self) -> str:
        """approved = target reached on pooled walk-forward signals and stable skill; otherwise shadow."""
        return "approved" if self.walk_forward["has_skill"] and self.gate["passed"] else "shadow"


def train_barrier_model(
    dataset: BarrierDataset,
    params: dict[str, Any] | None = None,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.15,
    folds: int = SKILL_GATE["folds"],
) -> BarrierTrainingResult:
    frame = dataset.frame.sort_values("timestamp").reset_index(drop=True)
    gap = interval_to_timedelta(dataset.interval) * dataset.max_periods
    split = chronological_split(frame["timestamp"], gap, validation_fraction, test_fraction)

    half = len(split.validation) // 2
    early_stop_all = split.validation[:half]
    calibration_all = split.validation[half + dataset.max_periods :]

    X = frame[dataset.feature_columns].to_numpy(dtype="float64")
    label = frame["label"].to_numpy(dtype="float64")
    outcome = frame["outcome"].to_numpy(dtype=object)
    durations = frame["resolve_periods"].to_numpy(dtype=int)

    train, early_stop, calibration = (_labeled(idx, label) for idx in (split.train, early_stop_all, calibration_all))
    if min(len(early_stop), len(calibration)) < MIN_FIT_ROWS:
        raise ValueError("not enough resolved validation rows for early stopping and calibration")

    model = _fit(X, label, train, early_stop, calibration, params)
    metrics = {
        "calibration": barrier_metrics(outcome[calibration_all], model.predict_proba_up(X[calibration_all]), durations[calibration_all]),
        "test": barrier_metrics(outcome[split.test], model.predict_proba_up(X[split.test]), durations[split.test]),
    }
    verification = walk_forward(dataset, folds, params)

    ts = frame["timestamp"]
    split_info = split.describe(ts)
    split_info["early_stopping"] = {"rows": len(early_stop_all), "start": ts.iloc[early_stop_all[0]].isoformat(), "end": ts.iloc[early_stop_all[-1]].isoformat()}
    split_info["calibration"] = {"rows": len(calibration_all), "start": ts.iloc[calibration_all[0]].isoformat(), "end": ts.iloc[calibration_all[-1]].isoformat()}
    split_info["fit_rows_resolved"] = {"train": int(len(train)), "early_stopping": int(len(early_stop)), "calibration": int(len(calibration))}

    logger.info(
        "Barrier model %s/%s: test accuracy %s, walk-forward skill %s, status %s",
        dataset.config.key,
        dataset.feature_set,
        metrics["test"]["accuracy"],
        verification["brier_skill_mean"],
        "approved" if verification["has_skill"] and verification["gate"]["passed"] else "shadow",
    )
    return BarrierTrainingResult(model, split_info, metrics, verification)
