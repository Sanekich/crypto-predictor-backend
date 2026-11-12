"""Direction classifier: probability that the price is higher after the horizon.

Raw classifier scores are not probabilities a user can trust. The model is
therefore calibrated on a separate time period, so that "75%" means "came
true in about 75% of comparable past cases".

Calibration uses Platt scaling (logistic regression on the log-odds of the raw
score). Isotonic regression was tried first and rejected: with few extreme
scores it overfits the tails and outputs "100%" for signals that came true in
only ~64% of test cases.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)

from app.ml.models.base import ReturnModel

DEFAULT_DIRECTION_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_child_weight": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "early_stopping_rounds": 100,
    "random_state": 42,
    "n_jobs": -1,
}


class DirectionModel(ReturnModel):
    """Interface: predict_proba_up() -> calibrated P(return > 0) per row."""

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_up(X)


class XGBoostDirectionModel(DirectionModel):
    name = "xgboost_direction"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = {**DEFAULT_DIRECTION_PARAMS, **(params or {})}
        self.model: XGBClassifier | None = None
        self.calibrator: LogisticRegression | None = None
        self.probability_shrinkage = 1.0

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBoostDirectionModel":
        """y is 1 for "up", 0 otherwise. Validation rows are used for early stopping only."""
        params = dict(self._params)
        eval_set = None
        if X_val is None or y_val is None:
            params.pop("early_stopping_rounds", None)
        else:
            eval_set = [(X_val, y_val)]
        self.model = XGBClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.calibrator = None
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "XGBoostDirectionModel":
        """Fit the probability calibration on a period NOT used for training or early stopping."""
        self.calibrator = LogisticRegression(C=1.0)
        self.calibrator.fit(_logit(self.raw_proba_up(X_cal)), np.asarray(y_cal, dtype=int))
        return self

    def set_probability_shrinkage(self, factor: float) -> "XGBoostDirectionModel":
        """Shrink calibrated probabilities toward 0.5 for unstable long horizons."""
        if not 0.0 < factor <= 1.0:
            raise ValueError("probability shrinkage must be in (0, 1]")
        self.probability_shrinkage = float(factor)
        return self

    def _require_model(self) -> XGBClassifier:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model

    def raw_proba_up(self, X: np.ndarray) -> np.ndarray:
        return self._require_model().predict_proba(X)[:, 1]

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        raw = self.raw_proba_up(X)
        if self.calibrator is None:
            return raw
        calibrated = self.calibrator.predict_proba(_logit(raw))[:, 1]
        return 0.5 + self.probability_shrinkage * (calibrated - 0.5)

    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def training_info(self) -> dict[str, Any]:
        best = getattr(self._require_model(), "best_iteration", None)
        return {
            "best_iteration": int(best) if best is not None else None,
            "calibrated": self.calibrator is not None,
            "probability_shrinkage": self.probability_shrinkage,
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._require_model().save_model(directory / "model.json")
        (directory / "params.json").write_text(json.dumps(self._params, indent=2), encoding="utf-8")
        if self.calibrator is not None:
            joblib.dump(self.calibrator, directory / "calibrator.joblib")
        (directory / "probability_shrinkage.json").write_text(str(self.probability_shrinkage), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "XGBoostDirectionModel":
        instance = cls(json.loads((directory / "params.json").read_text(encoding="utf-8")))
        instance.model = XGBClassifier()
        instance.model.load_model(directory / "model.json")
        calibrator_path = directory / "calibrator.joblib"
        if calibrator_path.exists():
            instance.calibrator = joblib.load(calibrator_path)
        shrinkage_path = directory / "probability_shrinkage.json"
        if shrinkage_path.exists():
            instance.probability_shrinkage = float(shrinkage_path.read_text(encoding="utf-8"))
        return instance
