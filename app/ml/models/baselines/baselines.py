"""Simple reference models. An ML model is only useful if it beats these."""

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.models.base import ReturnModel


class ZeroReturnBaseline(ReturnModel):
    """Price stays the same: predicted return is always 0 (the "last price" forecast)."""

    name = "zero_return"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ZeroReturnBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


class HistoricalMeanBaseline(ReturnModel):
    """Always predicts the average return observed in the training period."""

    name = "historical_mean"

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "HistoricalMeanBaseline":
        self.mean_ = float(np.mean(y_train))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("model is not fitted")
        return np.full(len(X), self.mean_)

    def training_info(self) -> dict[str, Any]:
        return {"mean": self.mean_}


class RidgeBaseline(ReturnModel):
    """Linear regression with L2 regularisation. The scaler is fitted on training rows only."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "RidgeBaseline":
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def params(self) -> dict[str, Any]:
        return {"alpha": self.alpha}
