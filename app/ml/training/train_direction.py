"""Train and verify a direction model (stage 1 of docs/direction_prediction_plan.md).

Time periods, oldest to newest:

    | train | gap | early stopping | gap | calibration | gap | test |

- the classifier is fitted on train, the number of trees is chosen on the
  early-stopping period;
- probabilities are calibrated on a later, separate period;
- the test period is used once, to decide whether the model is approved.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.ml.evaluation.metrics import direction_metrics
from app.ml.evaluation.splits import TimeSplit, chronological_split
from app.ml.evaluation.walk_forward import evaluate_direction_walk_forward
from app.ml.models.direction_model import XGBoostDirectionModel

logger = logging.getLogger(__name__)

SIGNAL_THRESHOLD = 0.75

# Direction is shown at all only if the model beats the base rate (Brier score)
# on most walk-forward folds. A hand-tuned probability shrinkage for 7d was tried
# and removed: walk-forward showed negative skill on all folds with or without it,
# i.e. it only reduced the damage of a model without signal.
SKILL_GATE = {"folds": 5, "min_positive_folds": 4}

# A model is "approved" (signals may be shown as reliable) only if, on the test
# period, its high-confidence signals came true often enough and on enough
# independent occasions (see plan, sections 3 and 7).
GATE = {
    "min_independent_signals": 77,
    "min_hit_rate": 0.75,
    "min_hit_rate_lower_95": 0.70,
}


def evaluate_gate(test_metrics: dict[str, Any], threshold: float = SIGNAL_THRESHOLD) -> dict[str, Any]:
    stats = test_metrics["signals"][f"{threshold:.2f}"]
    n = stats["independent_signals"]
    hit = stats["independent_hit_rate"]
    low = stats["hit_rate_lower_95"]
    reasons = []
    if n < GATE["min_independent_signals"]:
        reasons.append(f"only {n} independent signals >= {threshold:.0%} (need {GATE['min_independent_signals']})")
    if hit is not None and hit < GATE["min_hit_rate"]:
        reasons.append(f"hit rate {hit:.1%} below {GATE['min_hit_rate']:.0%}")
    if low is not None and low < GATE["min_hit_rate_lower_95"]:
        reasons.append(f"95% lower bound {low:.1%} below {GATE['min_hit_rate_lower_95']:.0%}")
    return {"passed": not reasons, "threshold": threshold, "criteria": GATE, "reasons": reasons}


@dataclass
class DirectionTrainingResult:
    model: XGBoostDirectionModel
    split: TimeSplit
    split_info: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    gate: dict[str, Any]

    @property
    def status(self) -> str:
        return "approved" if self.gate["passed"] else "shadow"


def train_direction_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon_periods: int,
    gap: pd.Timedelta,
    params: dict[str, Any] | None = None,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.15,
) -> DirectionTrainingResult:
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    split = chronological_split(frame["timestamp"], gap, validation_fraction, test_fraction)

    # Validation is divided into early stopping and calibration, with a gap between them
    half = len(split.validation) // 2
    early_stop = split.validation[:half]
    # All rows are used: subsampling to independent rows left ~18 points for 7d and
    # gave worse walk-forward skill for 24h; overlap does not bias a Platt fit.
    calibration = split.validation[half + horizon_periods :]
    if len(early_stop) == 0 or len(calibration) == 0:
        raise ValueError("not enough validation data for early stopping and calibration")

    X = frame[feature_columns].to_numpy(dtype="float64")
    y = frame[target_column].to_numpy(dtype="float64")

    model = XGBoostDirectionModel(params)
    model.fit(X[split.train], y[split.train], X[early_stop], y[early_stop])
    model.calibrate(X[calibration], y[calibration])

    metrics = {
        "calibration": direction_metrics(y[calibration], model.predict_proba_up(X[calibration]), horizon_periods),
        "test": direction_metrics(y[split.test], model.predict_proba_up(X[split.test]), horizon_periods),
    }
    gate = evaluate_gate(metrics["test"])

    ts = frame["timestamp"]
    split_info = split.describe(ts)
    split_info["early_stopping"] = {"rows": len(early_stop), "start": ts.iloc[early_stop[0]].isoformat(), "end": ts.iloc[early_stop[-1]].isoformat()}
    split_info["calibration"] = {"rows": len(calibration), "start": ts.iloc[calibration[0]].isoformat(), "end": ts.iloc[calibration[-1]].isoformat()}

    logger.info(
        "Direction model: test accuracy %.3f (always-up %.3f, majority %.3f), status %s",
        metrics["test"]["accuracy"],
        metrics["test"]["always_up_accuracy"],
        metrics["test"]["majority_baseline_accuracy"],
        "approved" if gate["passed"] else "shadow",
    )
    return DirectionTrainingResult(model, split, split_info, metrics, gate)


def walk_forward_skill(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon_periods: int,
    folds: int = SKILL_GATE["folds"],
) -> dict[str, Any]:
    """Brier skill vs the base rate on each expanding fold; has_skill decides if direction is shown."""
    results = evaluate_direction_walk_forward(frame, feature_columns, target_column, horizon_periods, folds)
    skills = [1 - r["brier"] / r["brier_base_rate"] for r in results]
    positive = sum(s > 0 for s in skills)
    return {
        "folds": len(skills),
        "brier_skill_by_fold": skills,
        "brier_skill_mean": float(np.mean(skills)) if skills else None,
        "positive_folds": positive,
        "accuracy_by_fold": [r["accuracy"] for r in results],
        "has_skill": bool(skills) and float(np.mean(skills)) > 0 and positive >= SKILL_GATE["min_positive_folds"],
        "criteria": SKILL_GATE,
    }


def to_up_label(direction: pd.Series) -> np.ndarray:
    return direction.to_numpy(dtype="float64")
