"""Features + barrier outcomes for training the new-logic models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.derivatives.store import DerivativesStore
from app.data.processing.resampling import reindex_to_grid
from app.data.storage import MarketDataStorage
from app.features.derivatives import GROUP_DATASET
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION, FeatureConfig, build_features, feature_columns
from app.ml.barrier.configs import FEATURE_SETS, BarrierConfig
from app.ml.barrier.labels import OUTCOMES, barrier_outcomes

OUTCOME_COLUMNS = ("outcome", "label", "resolve_periods")


@dataclass
class BarrierDataset:
    """Rows: one per closed candle with complete features and a known outcome.

    `label` is NaN for rows whose outcome is "none" or "both": they are not used for
    fitting, but they stay in the frame so that test metrics count how often a
    confident signal was followed by no significant move at all.
    """

    frame: pd.DataFrame
    feature_columns: list[str]
    config: BarrierConfig
    feature_set: str
    interval: str
    max_periods: int
    feature_config: FeatureConfig
    rows_before_filter: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        ts = self.frame["timestamp"]
        counts = self.frame["outcome"].value_counts()
        total = max(len(self.frame), 1)
        return {
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "interval": self.interval,
            "barrier": self.config.describe(),
            "max_periods": self.max_periods,
            "feature_set": self.feature_set,
            "rows": len(self.frame),
            "rows_dropped": self.rows_before_filter - len(self.frame),
            "labeled_rows": int(self.frame["label"].notna().sum()),
            "outcome_shares": {name: float(counts.get(name, 0)) / total for name in OUTCOMES if name != "unknown"},
            "first_timestamp": ts.iloc[0].isoformat() if len(ts) else None,
            "last_timestamp": ts.iloc[-1].isoformat() if len(ts) else None,
            "feature_columns": self.feature_columns,
            "feature_config": self.feature_config.to_dict(),
            **self.metadata,
        }


def build_barrier_dataset(
    ohlcv: pd.DataFrame,
    interval: str,
    config: BarrierConfig,
    feature_set: str = "candles",
    derivatives_data: dict[str, pd.DataFrame] | None = None,
) -> BarrierDataset:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {sorted(FEATURE_SETS)}")
    feature_config = FeatureConfig(derivatives=FEATURE_SETS[feature_set])
    features = build_features(ohlcv, interval, feature_config, derivatives_data)
    columns = feature_columns(features)

    max_periods = config.max_periods(interval)
    grid = reindex_to_grid(ohlcv, interval)
    outcomes = barrier_outcomes(grid, config.threshold, max_periods)
    frame = pd.concat([features, outcomes], axis=1)

    rows_before = int(frame["close"].notna().sum())
    clean = frame.dropna(subset=columns)
    clean = clean[clean["outcome"] != "unknown"].reset_index(drop=True)
    return BarrierDataset(
        frame=clean,
        feature_columns=columns,
        config=config,
        feature_set=feature_set,
        interval=interval,
        max_periods=max_periods,
        feature_config=feature_config,
        rows_before_filter=rows_before,
    )


def load_market_data(
    data_dir: Path, provider: str, symbol: str, interval: str, feature_set: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Processed candles plus the derivatives history the feature set needs."""
    ohlcv = MarketDataStorage(data_dir).load_processed(provider, symbol, interval)
    if ohlcv is None:
        raise FileNotFoundError("No processed candles. Run scripts/download_data.py first.")
    derivatives = DerivativesStore(data_dir).load_all(symbol)
    needed = {GROUP_DATASET[g] for g in FEATURE_SETS[feature_set]}
    missing = sorted(needed - set(derivatives))
    if missing:
        raise FileNotFoundError(
            f"Derivatives history missing for {', '.join(missing)}. Run scripts/download_derivatives.py first."
        )
    return ohlcv, derivatives
