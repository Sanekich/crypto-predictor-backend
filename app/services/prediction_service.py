"""Builds forecasts: latest closed candles -> features -> saved model -> response.

Rules (architecture doc, section 12):
- no training inside a request; models come from the registry;
- the model is used only with the same feature pipeline version and config;
- stale or incomplete data produce an error, never a made-up forecast.
"""

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import (
    DataStaleError,
    ModelIncompatibleError,
    ModelNotAvailableError,
)
from app.data.intervals import interval_to_timedelta
from app.features.feature_pipeline import FeatureConfig
from app.ml.models.base import ReturnModel
from app.ml.models.direction_model import DirectionModel
from app.ml.models.quantile_model import QuantileModel
from app.ml.registry.model_registry import ModelRegistry
from app.schemas.prediction import (
    AssetInfo,
    DirectionForecast,
    ModelQuality,
    PredictionResponse,
    PriceRange,
)
from app.services.features_service import FEATURE_HISTORY_ROWS, check_compatibility, latest_features
from app.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

RANGE_MODEL_TYPE = "xgboost_quantile"
DIRECTION_MODEL_TYPE = "xgboost_direction"

ASSET_NAMES = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "SOLUSDT": "Solana"}

__all__ = ["FEATURE_HISTORY_ROWS", "PredictionService"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionService:
    def __init__(
        self,
        registry: ModelRegistry,
        market_data: MarketDataService,
        interval: str = "1h",
        clock: Callable[[], datetime] = _utc_now,
        max_staleness_intervals: int = 3,
    ) -> None:
        self._registry = registry
        self._market_data = market_data
        self._interval = interval
        self._clock = clock
        self._max_staleness = interval_to_timedelta(interval) * max_staleness_intervals
        self._feature_config = FeatureConfig()
        self._models: dict[str, ReturnModel] = {}
        self._predictions: dict[tuple[str, pd.Timestamp], PredictionResponse] = {}
        self._lock = threading.Lock()

    # --- public API -----------------------------------------------------------

    def list_assets(self) -> list[AssetInfo]:
        horizons: dict[str, set[str]] = {}
        for meta in self._registry.list():
            if meta.get("model_type") == RANGE_MODEL_TYPE and meta.get("interval") == self._interval:
                horizons.setdefault(meta["symbol"], set()).add(meta["horizon"])
        return [
            AssetInfo(
                symbol=symbol,
                name=ASSET_NAMES.get(symbol, symbol),
                timeframe=self._interval,
                horizons=sorted(values, key=lambda h: pd.Timedelta(h.replace("d", "D"))),
            )
            for symbol, values in sorted(horizons.items())
        ]

    def predict(self, symbol: str, horizon: str) -> PredictionResponse:
        symbol = symbol.upper()
        meta = self._registry.latest(RANGE_MODEL_TYPE, symbol, self._interval, horizon)
        if meta is None:
            raise ModelNotAvailableError(f"No trained range model for {symbol} {self._interval} horizon {horizon}.")
        check_compatibility(meta, self._feature_config)

        ohlcv = self._market_data.get_ohlcv(symbol, self._interval)
        step = interval_to_timedelta(self._interval)
        last_open = ohlcv["timestamp"].iloc[-1]
        data_as_of = last_open + step
        now = self._clock()
        if pd.Timestamp(now) - data_as_of > self._max_staleness:
            raise DataStaleError(f"Latest closed candle is from {data_as_of.isoformat()}; data is too old for a forecast.")

        cache_key = (meta["model_id"], last_open)
        with self._lock:
            cached = self._predictions.get(cache_key)
        if cached is not None:
            return cached

        x = latest_features(ohlcv, self._interval, meta["feature_columns"], self._feature_config)
        model = self._load_model(meta["model_id"])
        quantile_returns = model.predict_quantiles(x)[0]
        quantiles = list(model.quantiles)

        current_price = float(ohlcv["close"].iloc[-1])
        median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
        lower, median, upper = quantile_returns[0], quantile_returns[median_idx], quantile_returns[-1]

        response = PredictionResponse(
            symbol=symbol,
            timeframe=self._interval,
            horizon=horizon,
            generated_at=now,
            data_as_of=data_as_of.to_pydatetime(),
            target_time=(data_as_of + step * meta["horizon_periods"]).to_pydatetime(),
            model_version=meta["model_id"],
            current_price=current_price,
            median_return=float(median),
            median_price=current_price * (1 + float(median)),
            range=PriceRange(
                probability=round(quantiles[-1] - quantiles[0], 6),
                lower_quantile=quantiles[0],
                upper_quantile=quantiles[-1],
                lower_return=float(lower),
                upper_return=float(upper),
                lower_price=current_price * (1 + float(lower)),
                upper_price=current_price * (1 + float(upper)),
            ),
            direction=self._direction(symbol, horizon, ohlcv),
            quality=self._quality(meta),
        )
        with self._lock:
            self._predictions = {k: v for k, v in self._predictions.items() if k[0] != meta["model_id"]}
            self._predictions[cache_key] = response
        return response

    # --- internals ------------------------------------------------------------

    def _load_model(self, model_id: str) -> QuantileModel:
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            loaded, _ = self._registry.load(model_id)
            if not isinstance(loaded, QuantileModel):
                raise ModelIncompatibleError(f"Model {model_id} is not a range model")
            model = loaded
            with self._lock:
                self._models[model_id] = model
            logger.info("Loaded model %s", model_id)
        return model

    def _load_direction_model(self, model_id: str) -> DirectionModel:
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            loaded, _ = self._registry.load(model_id)
            if not isinstance(loaded, DirectionModel):
                raise ModelIncompatibleError(f"Model {model_id} is not a direction model")
            model = loaded
            with self._lock:
                self._models[model_id] = model
            logger.info("Loaded direction model %s", model_id)
        return model

    def _direction(self, symbol: str, horizon: str, ohlcv: pd.DataFrame) -> DirectionForecast:
        meta = self._registry.latest(DIRECTION_MODEL_TYPE, symbol, self._interval, horizon)
        if meta is None:
            return DirectionForecast(available=False, reason="NO_MODEL")
        check_compatibility(meta, self._feature_config)
        features = latest_features(ohlcv, self._interval, meta["feature_columns"], self._feature_config)
        model = self._load_direction_model(meta["model_id"])
        p_up = float(model.predict_proba_up(features)[0])
        confidence = max(p_up, 1 - p_up)
        status = meta.get("status", "shadow")
        return DirectionForecast(
            available=True,
            up_probability=p_up,
            direction="up" if p_up >= 0.5 else "down",
            confidence=confidence,
            status=status,
            model_version=meta["model_id"],
            reason=None if status == "approved" else "SHADOW",
        )

    @staticmethod
    def _quality(meta: dict[str, Any]) -> ModelQuality:
        test = meta["metrics"]["test"]
        return ModelQuality(
            test_period_start=meta["split"]["test"]["start"],
            test_period_end=meta["split"]["test"]["end"],
            nominal_coverage=test["nominal_coverage"],
            test_coverage=test["coverage"],
            improvement_vs_baseline=meta["improvement_vs_baseline"]["pinball"]["test"],
        )
