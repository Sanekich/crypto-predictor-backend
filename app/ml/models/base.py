"""Common interface for models that predict future returns."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ReturnModel(ABC):
    """fit() on training rows, optionally using validation rows (early stopping);
    predict() returns the forecast return for each input row."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "ReturnModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def params(self) -> dict[str, Any]:
        return {}

    def training_info(self) -> dict[str, Any]:
        return {}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")

    @classmethod
    def load(cls, directory: Path) -> "ReturnModel":
        model = joblib.load(directory / "model.joblib")
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model
