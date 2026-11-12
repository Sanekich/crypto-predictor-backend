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
from app.ml.evaluation.metrics import independent_sample_indices
from app.ml.evaluation.splits import TimeSplit, chronological_split
from app.ml.models.direction_model import XGBoostDirectionModel

logger = logging.getLogger(__name__)

SIGNAL_THRESHOLD = 0.75
WEEKLY_PROBABILITY_SHRINKAGE = 0.35

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
    calibration = split.validation[half + horizon_periods :]
    calibration = independent_sample_indices(calibration, horizon_periods)
    if len(early_stop) == 0 or len(calibration) == 0:
        raise ValueError("not enough validation data for early stopping and calibration")

    X = frame[feature_columns].to_numpy(dtype="float64")
    y = frame[target_column].to_numpy(dtype="float64")

    model = XGBoostDirectionModel(params)
    model.fit(X[split.train], y[split.train], X[early_stop], y[early_stop])
    model.calibrate(X[calibration], y[calibration])
    if horizon_periods >= 24 * 7:
        model.set_probability_shrinkage(WEEKLY_PROBABILITY_SHRINKAGE)

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
        "Direction model: test accuracy %.3f (always-up %.3f), status %s",
        metrics["test"]["accuracy"],
        metrics["test"]["always_up_accuracy"],
        "approved" if gate["passed"] else "shadow",
    )
    return DirectionTrainingResult(model, split, split_info, metrics, gate)


def to_up_label(direction: pd.Series) -> np.ndarray:
    return direction.to_numpy(dtype="float64")
