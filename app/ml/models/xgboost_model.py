"""XGBoost regressor for future returns."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from app.ml.models.base import ReturnModel

# Conservative defaults for noisy financial data: shallow trees, small learning
# rate, large leaves and subsampling to limit overfitting. The number of trees
# is chosen by early stopping on the validation period.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "early_stopping_rounds": 100,
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostReturnModel(ReturnModel):
    name = "xgboost"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: XGBRegressor | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBoostReturnModel":
        params = dict(self._params)
        if X_val is None or y_val is None:
            params.pop("early_stopping_rounds", None)
            eval_set = None
        else:
            eval_set = [(X_val, y_val)]
        self.model = XGBRegressor(**params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self

    def _require_model(self) -> XGBRegressor:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        # With early stopping, predict() uses the best iteration automatically
        return self._require_model().predict(X)

    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def training_info(self) -> dict[str, Any]:
        model = self._require_model()
        best = getattr(model, "best_iteration", None)
        return {"best_iteration": int(best) if best is not None else None}

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Total gain per feature, normalised to sum to 1, sorted descending."""
        scores = self._require_model().get_booster().get_score(importance_type="total_gain")
        # Booster names features f0, f1, ... when trained on numpy arrays
        named = {feature_names[int(key[1:])]: value for key, value in scores.items()}
        total = sum(named.values()) or 1.0
        return dict(sorted(((k, v / total) for k, v in named.items()), key=lambda kv: -kv[1]))

    def save(self, directory: Path) -> None:
        """XGBoost's own JSON format: portable across library versions, unlike pickle."""
        directory.mkdir(parents=True, exist_ok=True)
        self._require_model().save_model(directory / "model.json")
        (directory / "params.json").write_text(json.dumps(self._params, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "XGBoostReturnModel":
        params = json.loads((directory / "params.json").read_text(encoding="utf-8"))
        instance = cls(params)
        instance.model = XGBRegressor()
        instance.model.load_model(directory / "model.json")
        return instance
