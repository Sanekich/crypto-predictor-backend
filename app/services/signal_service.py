"""Short-horizon direction signals (1-4 hours) with verification status.

Every horizon with a trained direction model is reported, including its
current calibrated probability. Signals at or above the model's threshold
(75%) are written to the journal so their real hit rate can be shown.
"""

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.exceptions import DataStaleError, ModelIncompatibleError
from app.data.intervals import interval_to_timedelta
from app.features.feature_pipeline import FeatureConfig
from app.ml.models.direction_model import DirectionModel
from app.ml.registry.model_registry import ModelRegistry
from app.schemas.signals import (
    HorizonSignal,
    JournalEntry,
    LiveStats,
    SignalModelQuality,
    SignalsResponse,
)
from app.services.features_service import check_compatibility, latest_features
from app.services.market_data_service import MarketDataService
from app.services.signal_journal import SignalJournal

logger = logging.getLogger(__name__)

DIRECTION_MODEL_TYPE = "xgboost_direction"
DEFAULT_THRESHOLD = 0.75


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _horizon_sort_key(horizon: str) -> pd.Timedelta:
    return pd.Timedelta(horizon.replace("d", "D"))


class SignalService:
    def __init__(
        self,
        registry: ModelRegistry,
        market_data: MarketDataService,
        journal: SignalJournal,
        interval: str = "1h",
        clock: Callable[[], datetime] = _utc_now,
        max_staleness_intervals: int = 3,
    ) -> None:
        self._registry = registry
        self._market_data = market_data
        self._journal = journal
        self._interval = interval
        self._clock = clock
        self._max_staleness = interval_to_timedelta(interval) * max_staleness_intervals
        self._feature_config = FeatureConfig()
        self._models: dict[str, DirectionModel] = {}
        self._lock = threading.Lock()

    def _direction_models(self, symbol: str) -> list[dict[str, Any]]:
        horizons = {
            m["horizon"]
            for m in self._registry.list()
            if m.get("model_type") == DIRECTION_MODEL_TYPE and m.get("symbol") == symbol and m.get("interval") == self._interval
        }
        metas = [self._registry.latest(DIRECTION_MODEL_TYPE, symbol, self._interval, h) for h in horizons]
        return sorted((m for m in metas if m), key=lambda m: _horizon_sort_key(m["horizon"]))

    def _load(self, model_id: str) -> DirectionModel:
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            loaded, _ = self._registry.load(model_id)
            if not isinstance(loaded, DirectionModel):
                raise ModelIncompatibleError(f"Model {model_id} is not a direction model")
            model = loaded
            with self._lock:
                self._models[model_id] = model
        return model

    def get_signals(self, symbol: str, threshold: float = DEFAULT_THRESHOLD) -> SignalsResponse:
        symbol = symbol.upper()
        ohlcv = self._market_data.get_ohlcv(symbol, self._interval)
        step = interval_to_timedelta(self._interval)
        data_as_of = ohlcv["timestamp"].iloc[-1] + step
        now = self._clock()
        if pd.Timestamp(now) - data_as_of > self._max_staleness:
            raise DataStaleError(f"Latest closed candle is from {data_as_of.isoformat()}; data is too old for signals.")

        price = float(ohlcv["close"].iloc[-1])
        horizons: list[HorizonSignal] = []
        features_cache: dict[tuple[str, ...], Any] = {}

        for meta in self._direction_models(symbol):
            check_compatibility(meta, self._feature_config)
            columns = tuple(meta["feature_columns"])
            if columns not in features_cache:
                features_cache[columns] = latest_features(ohlcv, self._interval, list(columns), self._feature_config)
            p_up = float(self._load(meta["model_id"]).predict_proba_up(features_cache[columns])[0])
            confidence = max(p_up, 1 - p_up)
            direction = "up" if p_up >= 0.5 else "down"
            target_time = data_as_of + step * meta["horizon_periods"]
            model_threshold = float(meta.get("signal_threshold", DEFAULT_THRESHOLD))

            if confidence >= model_threshold:
                self._journal.record(
                    symbol,
                    {
                        "horizon": meta["horizon"],
                        "signal_time": data_as_of,
                        "target_time": target_time,
                        "direction": direction,
                        "p_up": p_up,
                        "confidence": confidence,
                        "price_at_signal": price,
                        "model_version": meta["model_id"],
                        "model_status": meta.get("status", "shadow"),
                    },
                )
                logger.info("Signal %s %s %s %.1f%%", symbol, meta["horizon"], direction, confidence * 100)

            horizons.append(
                HorizonSignal(
                    horizon=meta["horizon"],
                    model_version=meta["model_id"],
                    status="approved" if meta.get("status") == "approved" else "shadow",
                    status_reasons=meta.get("gate", {}).get("reasons", []),
                    p_up=p_up,
                    direction=direction,
                    confidence=confidence,
                    is_signal=confidence >= threshold,
                    target_time=target_time.to_pydatetime(),
                    quality=self._quality(meta, model_threshold),
                )
            )

        history = self._journal.with_outcomes(symbol, ohlcv, self._interval)
        entries = [
            JournalEntry(
                horizon=row.horizon,
                signal_time=row.signal_time,
                target_time=row.target_time,
                direction=row.direction,
                confidence=row.confidence,
                price_at_signal=row.price_at_signal,
                outcome_price=None if pd.isna(row.outcome_price) else float(row.outcome_price),
                correct=None if pd.isna(row.correct) else bool(row.correct),
                model_status=row.model_status,
            )
            for row in history.itertuples()
        ] if not history.empty else []

        resolved = [e for e in entries if e.correct is not None]
        correct = sum(e.correct for e in resolved)
        return SignalsResponse(
            symbol=symbol,
            generated_at=now,
            data_as_of=data_as_of.to_pydatetime(),
            threshold=threshold,
            horizons=horizons,
            history=entries,
            live=LiveStats(
                signals=len(entries),
                resolved=len(resolved),
                correct=correct,
                hit_rate=correct / len(resolved) if resolved else None,
            ),
        )

    @staticmethod
    def _quality(meta: dict[str, Any], threshold: float) -> SignalModelQuality:
        test = meta["metrics"]["test"]
        stats = test["signals"].get(f"{threshold:.2f}", {})
        return SignalModelQuality(
            test_period_start=meta["split"]["test"]["start"],
            test_period_end=meta["split"]["test"]["end"],
            test_accuracy=test["accuracy"],
            always_up_accuracy=test["always_up_accuracy"],
            max_confidence=test["max_confidence"],
            signals_at_threshold=stats.get("independent_signals", 0),
            hit_rate=stats.get("independent_hit_rate"),
            hit_rate_lower_95=stats.get("hit_rate_lower_95"),
        )
