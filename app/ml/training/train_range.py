"""Train range (quantile) models on a chronological split and compare them.

As in train.py, selection uses validation only; test is for reporting.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.ml.evaluation.metrics import interval_metrics
from app.ml.evaluation.splits import TimeSplit, chronological_split
from app.ml.models.quantile_model import (
    DEFAULT_QUANTILES,
    HistoricalQuantileBaseline,
    QuantileModel,
    XGBoostQuantileModel,
)

logger = logging.getLogger(__name__)


def default_range_models(quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> list[QuantileModel]:
    return [HistoricalQuantileBaseline(quantiles), XGBoostQuantileModel(quantiles)]


@dataclass
class RangeEvaluation:
    model: QuantileModel
    metrics: dict[str, dict[str, Any]]

    @property
    def name(self) -> str:
        return self.model.name


@dataclass
class RangeTrainingResult:
    split: TimeSplit
    split_info: dict[str, Any]
    evaluations: list[RangeEvaluation] = field(default_factory=list)

    def get(self, name: str) -> RangeEvaluation:
        return next(e for e in self.evaluations if e.name == name)


def train_range_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    gap: pd.Timedelta,
    models: list[QuantileModel] | None = None,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> RangeTrainingResult:
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    split = chronological_split(frame["timestamp"], gap, validation_fraction, test_fraction)
    result = RangeTrainingResult(split=split, split_info=split.describe(frame["timestamp"]))

    X = frame[feature_columns].to_numpy(dtype="float64")
    y = frame[target_column].to_numpy(dtype="float64")

    for model in models if models is not None else default_range_models():
        logger.info("Training %s", model.name)
        model.fit(X[split.train], y[split.train], X[split.validation], y[split.validation])
        metrics = {
            name: interval_metrics(y[idx], model.predict_quantiles(X[idx]), model.quantiles)
            for name, idx in split.items()
        }
        result.evaluations.append(RangeEvaluation(model=model, metrics=metrics))
        logger.info(
            "%s: validation coverage %.3f, width %.0f bps",
            model.name,
            metrics["validation"]["coverage"],
            metrics["validation"]["mean_width_bps"],
        )
    return result
