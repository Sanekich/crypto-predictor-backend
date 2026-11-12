"""Experimental three-class direction model: down / neutral / up."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from app.ml.models.base import ReturnModel


DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "early_stopping_rounds": 100,
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostNeutralDirectionModel(ReturnModel):
    """Class order is 0=down, 1=neutral, 2=up."""

    name = "xgboost_neutral_direction"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: XGBClassifier | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None) -> "XGBoostNeutralDirectionModel":
        params = dict(self._params)
        eval_set = None
        if X_val is None or y_val is None:
            params.pop("early_stopping_rounds", None)
        else:
            eval_set = [(X_val, y_val)]
        self.model = XGBClassifier(**params)
        self.model.fit(X_train, y_train.astype("int64"), sample_weight=sample_weight, eval_set=eval_set, verbose=False)
        return self

    def _require_model(self) -> XGBClassifier:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model

    def predict_proba_classes(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._require_model().predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._require_model().predict(X).astype("int64")

    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def training_info(self) -> dict[str, Any]:
        best = getattr(self._require_model(), "best_iteration", None)
        return {"best_iteration": int(best) if best is not None else None}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._require_model().save_model(directory / "model.json")
        (directory / "params.json").write_text(json.dumps(self._params, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "XGBoostNeutralDirectionModel":
        instance = cls(json.loads((directory / "params.json").read_text(encoding="utf-8")))
        instance.model = XGBClassifier()
        instance.model.load_model(directory / "model.json")
        return instance
