"""Feature-set loading for independent direction-model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from app.data.storage import MarketDataStorage
from app.features.feature_pipeline import FeatureDataset, build_dataset
from app.features.targets import future_direction, future_return, horizon_to_periods

FeatureSet = Literal["v1", "v2_funding_basis"]

FEATURE_SET_LABELS: dict[str, str] = {
    "v1": "OHLCV",
    "v2_funding_basis": "OHLCV + funding + basis",
}


def _v2_candidates(data_dir: Path, provider: str, symbol: str, interval: str, horizon: str) -> list[Path]:
    base = data_dir / "features" / provider
    stem = f"{symbol.upper()}_{interval}_h{horizon}"
    return [
        base / f"{stem}_v2.parquet",
        base / f"{stem}_v2_funding_basis.parquet",
        base / f"{stem}_derivatives.parquet",
    ]


def load_direction_dataset(
    data_dir: Path,
    provider: str,
    symbol: str,
    interval: str,
    horizon: str,
    feature_set: FeatureSet = "v1",
) -> FeatureDataset:
    """Load a causal direction dataset for the selected independent experiment.

    v2 deliberately requires a prepared, timestamp-aligned parquet file. It must
    contain the normal v1 feature columns plus ``funding_rate`` and
    ``basis_mean``. We do not silently forward-fill or fabricate derivatives
    history because that could create leakage or make a partial live sample look
    like a valid historical training set.
    """
    storage = MarketDataStorage(data_dir)
    ohlcv = storage.load_processed(provider, symbol, interval)
    if ohlcv is None:
        raise FileNotFoundError("No processed data. Run scripts/download_data.py first.")
    if feature_set == "v1":
        return build_dataset(ohlcv, interval, horizon)
    if feature_set != "v2_funding_basis":
        raise ValueError(f"Unsupported feature set: {feature_set}")

    path = next((candidate for candidate in _v2_candidates(data_dir, provider, symbol, interval, horizon) if candidate.exists()), None)
    if path is None:
        expected = ", ".join(str(p.relative_to(data_dir)) for p in _v2_candidates(data_dir, provider, symbol, interval, horizon))
        raise FileNotFoundError(
            "v2 dataset is not prepared. Create a timestamp-aligned parquet file at one of: "
            f"{expected}; it must include funding_rate and basis_mean."
        )

    frame = pd.read_parquet(path).copy()
    if "timestamp" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"v2 dataset {path.name} must contain timestamp and close columns")
    required = {"funding_rate", "basis_mean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v2 dataset {path.name} is missing required columns: {', '.join(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    periods = horizon_to_periods(horizon, interval)
    target_return = f"target_return_{horizon}"
    target_direction = f"target_direction_{horizon}"
    frame[target_return] = future_return(frame["close"], periods)
    frame[target_direction] = future_direction(frame[target_return])
    non_features = {"timestamp", "close", target_return, target_direction}
    columns = [column for column in frame.columns if column not in non_features]
    clean = frame.dropna(subset=[*columns, target_return]).reset_index(drop=True)
    if len(clean) < 1000:
        raise ValueError(f"v2 dataset has only {len(clean)} usable rows; collect/prepare more aligned history before training")
    from app.features.feature_pipeline import FeatureConfig

    return FeatureDataset(
        frame=clean,
        feature_columns=columns,
        target_return_column=target_return,
        target_direction_column=target_direction,
        horizon=horizon,
        horizon_periods=periods,
        interval=interval,
        config=FeatureConfig(),
        rows_before_dropna=len(frame),
        metadata={"feature_set": feature_set, "source_file": str(path.relative_to(data_dir))},
    )
