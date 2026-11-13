"""Feature-set loading for independent direction-model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from app.data.derivatives.store import DerivativesStore
from app.data.storage import MarketDataStorage
from app.features.feature_pipeline import FeatureConfig, FeatureDataset, build_dataset

FeatureSet = Literal["v1", "v2_funding_basis"]

FEATURE_SET_LABELS: dict[str, str] = {
    "v1": "OHLCV",
    "v2_funding_basis": "OHLCV + funding + basis",
}


def load_direction_dataset(
    data_dir: Path,
    provider: str,
    symbol: str,
    interval: str,
    horizon: str,
    feature_set: FeatureSet = "v1",
) -> FeatureDataset:
    """Build a causal direction dataset from the selected repository feature set.

    v2 uses the existing DerivativesStore and feature pipeline. The derivative
    feature builder performs publication-time as-of joins and leaves outages as
    NaN; build_dataset then drops unusable rows. This avoids silently using
    future values or treating a partial live sample as historical data.
    """
    storage = MarketDataStorage(data_dir)
    ohlcv = storage.load_processed(provider, symbol, interval)
    if ohlcv is None:
        raise FileNotFoundError("No processed data. Run scripts/download_data.py first.")
    if feature_set == "v1":
        return build_dataset(ohlcv, interval, horizon)
    if feature_set != "v2_funding_basis":
        raise ValueError(f"Unsupported feature set: {feature_set}")

    derivatives = DerivativesStore(data_dir).load_all(symbol)
    required = {"funding", "premium"}
    missing = sorted(required - set(derivatives))
    if missing:
        raise FileNotFoundError(
            "v2 derivatives history is incomplete; run scripts/download_derivatives.py first. "
            f"Missing datasets: {', '.join(missing)}"
        )
    dataset = build_dataset(
        ohlcv,
        interval,
        horizon,
        FeatureConfig(derivatives=("funding", "basis")),
        derivatives,
    )
    dataset.metadata.update(
        {
            "feature_set": feature_set,
            "derivatives_groups": ["funding", "basis"],
            "derivatives_source": "data/derivatives",
        }
    )
    return dataset
