"""Train several models on the same chronological split and compare them.

Model selection uses the validation period only. Test metrics are computed
for reporting and must not be used to tune models (architecture doc, 9.3).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.ml.evaluation.metrics import regression_metrics
from app.ml.evaluation.splits import TimeSplit, chronological_split
from app.ml.models.base import ReturnModel
from app.ml.models.baselines import HistoricalMeanBaseline, RidgeBaseline, ZeroReturnBaseline
from app.ml.models.xgboost_model import XGBoostReturnModel

logger = logging.getLogger(__name__)

SELECTION_METRIC = "rmse"


def default_models() -> list[ReturnModel]:
    return [ZeroReturnBaseline(), HistoricalMeanBaseline(), RidgeBaseline(), XGBoostReturnModel()]


@dataclass
class ModelEvaluation:
    model: ReturnModel
    metrics: dict[str, dict[str, Any]]

    @property
    def name(self) -> str:
        return self.model.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.model.params(),
            "training_info": self.model.training_info(),
            "metrics": self.metrics,
        }


@dataclass
class TrainingResult:
    split: TimeSplit
    split_info: dict[str, Any]
    evaluations: list[ModelEvaluation] = field(default_factory=list)

    def get(self, name: str) -> ModelEvaluation:
        return next(e for e in self.evaluations if e.name == name)

    @property
    def best_by_validation(self) -> ModelEvaluation:
        return min(self.evaluations, key=lambda e: e.metrics["validation"][SELECTION_METRIC])

    def best_baseline(self, exclude: str = "xgboost") -> ModelEvaluation:
        baselines = [e for e in self.evaluations if e.name != exclude]
        return min(baselines, key=lambda e: e.metrics["validation"][SELECTION_METRIC])

    def comparison(self) -> dict[str, Any]:
        return {e.name: e.to_dict() for e in self.evaluations}


def train_and_evaluate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    gap: pd.Timedelta,
    models: list[ReturnModel] | None = None,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> TrainingResult:
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    split = chronological_split(frame["timestamp"], gap, validation_fraction, test_fraction)
    result = TrainingResult(split=split, split_info=split.describe(frame["timestamp"]))

    X = frame[feature_columns].to_numpy(dtype="float64")
    y = frame[target_column].to_numpy(dtype="float64")
    X_train, y_train = X[split.train], y[split.train]
    X_val, y_val = X[split.validation], y[split.validation]

    for model in models if models is not None else default_models():
        logger.info("Training %s", model.name)
        model.fit(X_train, y_train, X_val, y_val)
        metrics = {name: regression_metrics(y[idx], model.predict(X[idx])) for name, idx in split.items()}
        result.evaluations.append(ModelEvaluation(model=model, metrics=metrics))
        logger.info(
            "%s: validation RMSE %.5f, direction accuracy %.3f",
            model.name,
            metrics["validation"]["rmse"],
            metrics["validation"]["direction_accuracy"],
        )

    return result
