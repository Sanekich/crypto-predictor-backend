"""Models that forecast a range (quantiles) of the future return.

A quantile model for levels (0.1, 0.5, 0.9) answers: "the return will be
below q0.1 in ~10% of cases, below q0.9 in ~90% of cases", i.e. an 80%
prediction interval. This is useful even when the direction is not
predictable, because volatility (the width of the range) is.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from app.ml.models.base import ReturnModel

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

DEFAULT_QUANTILE_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "early_stopping_rounds": 100,
    "random_state": 42,
    "n_jobs": -1,
}


class QuantileModel(ReturnModel):
    """Common interface: predict_quantiles() -> array of shape (n_rows, n_quantiles)."""

    quantiles: tuple[float, ...] = DEFAULT_QUANTILES

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point forecast: the median (or the quantile closest to it)."""
        idx = int(np.argmin([abs(q - 0.5) for q in self.quantiles]))
        return self.predict_quantiles(X)[:, idx]


class HistoricalQuantileBaseline(QuantileModel):
    """Same range for every row: empirical quantiles of the training period."""

    name = "historical_quantiles"

    def __init__(self, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> None:
        self.quantiles = tuple(quantiles)
        self.values_: np.ndarray | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "HistoricalQuantileBaseline":
        self.values_ = np.quantile(y_train, self.quantiles)
        return self

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        if self.values_ is None:
            raise RuntimeError("model is not fitted")
        return np.tile(self.values_, (len(X), 1))

    def params(self) -> dict[str, Any]:
        return {"quantiles": list(self.quantiles)}


class XGBoostQuantileModel(QuantileModel):
    """One XGBoost model fitted for several quantiles at once (reg:quantileerror)."""

    name = "xgboost_quantile"

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        params: dict[str, Any] | None = None,
    ) -> None:
        if list(quantiles) != sorted(quantiles) or not all(0 < q < 1 for q in quantiles):
            raise ValueError("quantiles must be sorted and inside (0, 1)")
        self.quantiles = tuple(quantiles)
        self._params = {**DEFAULT_QUANTILE_PARAMS, **(params or {})}
        self.model: XGBRegressor | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBoostQuantileModel":
        params = dict(self._params)
        eval_set = None
        if X_val is None or y_val is None:
            params.pop("early_stopping_rounds", None)
        else:
            eval_set = [(X_val, y_val)]
        self.model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.array(self.quantiles),
            **params,
        )
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self

    def _require_model(self) -> XGBRegressor:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self._require_model().predict(X)).reshape(len(X), len(self.quantiles))
        # Separately fitted quantiles can cross; sorting restores a valid order
        return np.sort(raw, axis=1)

    def params(self) -> dict[str, Any]:
        return {"quantiles": list(self.quantiles), **self._params}

    def training_info(self) -> dict[str, Any]:
        best = getattr(self._require_model(), "best_iteration", None)
        return {"best_iteration": int(best) if best is not None else None}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._require_model().save_model(directory / "model.json")
        payload = {"quantiles": list(self.quantiles), "params": self._params}
        (directory / "params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "XGBoostQuantileModel":
        payload = json.loads((directory / "params.json").read_text(encoding="utf-8"))
        instance = cls(tuple(payload["quantiles"]), payload["params"])
        instance.model = XGBRegressor()
        instance.model.load_model(directory / "model.json")
        return instance
