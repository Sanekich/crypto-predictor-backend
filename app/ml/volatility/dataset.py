"""Features + "big move" labels for the volatility models."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.data.processing.resampling import reindex_to_grid
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION, FeatureConfig, build_features, feature_columns
from app.ml.barrier.labels import barrier_outcomes
from app.ml.volatility.configs import FEATURE_SETS, VolatilityConfig

# A plain, well-known volatility measure used by the naive baseline
BASELINE_FEATURE = "volatility_24"


@dataclass
class VolatilityDataset:
    frame: pd.DataFrame  # timestamp, close, features, label (1 = big move), resolve_periods
    feature_columns: list[str]
    config: VolatilityConfig
    feature_set: str
    interval: str
    periods: int
    feature_config: FeatureConfig
    rows_before_filter: int = 0

    def describe(self) -> dict[str, Any]:
        ts = self.frame["timestamp"]
        return {
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "interval": self.interval,
            "config": self.config.describe(),
            "periods": self.periods,
            "feature_set": self.feature_set,
            "rows": len(self.frame),
            "rows_dropped": self.rows_before_filter - len(self.frame),
            "move_share": float(self.frame["label"].mean()) if len(ts) else None,
            "first_timestamp": ts.iloc[0].isoformat() if len(ts) else None,
            "last_timestamp": ts.iloc[-1].isoformat() if len(ts) else None,
        }


def build_volatility_dataset(
    ohlcv: pd.DataFrame,
    interval: str,
    config: VolatilityConfig,
    feature_set: str = "candles",
    derivatives_data: dict[str, pd.DataFrame] | None = None,
) -> VolatilityDataset:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {sorted(FEATURE_SETS)}")
    feature_config = FeatureConfig(derivatives=FEATURE_SETS[feature_set])
    features = build_features(ohlcv, interval, feature_config, derivatives_data)
    columns = feature_columns(features)
    periods = config.periods(interval)

    outcomes = barrier_outcomes(reindex_to_grid(ohlcv, interval), config.threshold, periods)
    frame = pd.concat([features, outcomes[["outcome", "resolve_periods"]]], axis=1)
    rows_before = int(frame["close"].notna().sum())
    # "unknown": the future is not fully known yet (end of data or missing candles)
    frame = frame.dropna(subset=columns)
    frame = frame[frame["outcome"] != "unknown"].copy()
    frame["label"] = (frame["outcome"] != "none").astype("float64")
    frame = frame.drop(columns=["outcome"]).reset_index(drop=True)
    return VolatilityDataset(frame, columns, config, feature_set, interval, periods, feature_config, rows_before)
