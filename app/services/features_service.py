"""Feature vector of the latest closed candle, shared by all inference services."""

from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import DataNotAvailableError, ModelIncompatibleError
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION, FeatureConfig, build_features

# Enough history for all rolling windows and for exponential averages to converge
FEATURE_HISTORY_ROWS = 2000


def check_compatibility(meta: dict[str, Any], config: FeatureConfig) -> None:
    """A model may only be used with the feature pipeline it was trained with."""
    if meta.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION:
        raise ModelIncompatibleError(
            f"Model {meta['model_id']} was trained with feature pipeline "
            f"{meta.get('feature_pipeline_version')}, current is {FEATURE_PIPELINE_VERSION}. Retrain the model."
        )
    try:
        stored = FeatureConfig.from_dict(meta.get("feature_config") or {})
    except (TypeError, ValueError):
        raise ModelIncompatibleError(f"Model {meta['model_id']} has an unreadable feature configuration.") from None
    if stored.derivatives:
        raise ModelIncompatibleError(
            f"Model {meta['model_id']} needs derivatives features ({', '.join(stored.derivatives)}), "
            "which the API does not compute at inference yet. Use it for offline comparison only."
        )
    if stored != config:
        raise ModelIncompatibleError(f"Model {meta['model_id']} uses a different feature configuration. Retrain the model.")


def latest_features(
    ohlcv: pd.DataFrame,
    interval: str,
    columns: list[str],
    config: FeatureConfig,
) -> np.ndarray:
    """Features of the last candle as a (1, n_features) array, in the model's column order."""
    frame = build_features(ohlcv.iloc[-FEATURE_HISTORY_ROWS:], interval, config)
    missing_columns = [c for c in columns if c not in frame.columns]
    if missing_columns:
        raise ModelIncompatibleError(f"Features missing from pipeline: {', '.join(missing_columns)}")

    last = frame.iloc[-1]
    if last["timestamp"] != ohlcv["timestamp"].iloc[-1]:
        raise DataNotAvailableError("Feature frame is not aligned with the latest candle.")
    values = last[columns].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        bad = [c for c, v in zip(columns, values) if not np.isfinite(v)]
        raise DataNotAvailableError(
            f"Not enough continuous recent data to compute features ({', '.join(bad[:5])}). "
            "This happens right after gaps in exchange data."
        )
    return values.reshape(1, -1)
