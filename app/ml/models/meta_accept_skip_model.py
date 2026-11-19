"""Small, serializable meta-model for accepting or skipping primary signals."""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.ml.models.base import ReturnModel


class MetaAcceptSkipModel(ReturnModel):
    name = "xgboost_meta_accept_skip"

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or ["primary_probability", "primary_confidence", "confidence_squared"]
        self.model: LogisticRegression | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None) -> "MetaAcceptSkipModel":
        self.model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)
        self.model.fit(X_train, y_train.astype(int))
        return self

    def _require_model(self) -> LogisticRegression:
        if self.model is None:
            raise RuntimeError("meta-model is not fitted")
        return self.model

    def predict_accept_proba(self, X: np.ndarray) -> np.ndarray:
        return self._require_model().predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_accept_proba(X)

    def params(self) -> dict[str, Any]:
        return {"C": 0.5, "class_weight": "balanced", "feature_names": self.feature_names}

    def training_info(self) -> dict[str, Any]:
        return {"model": "logistic_regression", "feature_names": self.feature_names}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._require_model(), directory / "model.joblib")
        (directory / "params.json").write_text(
            __import__("json").dumps(self.params(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "MetaAcceptSkipModel":
        params = __import__("json").loads((directory / "params.json").read_text(encoding="utf-8"))
        instance = cls(params.get("feature_names"))
        instance.model = joblib.load(directory / "model.joblib")
        return instance
