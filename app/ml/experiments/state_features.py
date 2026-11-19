"""Causal market-state features for isolated experiments.

Every feature at row t uses only values through row t. These are numeric state
signals, not claims about the cause of a move: a shock may be news, liquidation,
or another market event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name] if name in frame else pd.Series(np.nan, index=frame.index, dtype="float64")


def build_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return state/shock features aligned to a barrier feature frame."""
    r1 = _column(frame, "log_return_1")
    r24 = _column(frame, "log_return_24")
    vol24 = _column(frame, "volatility_24").abs()
    vol168 = _column(frame, "volatility_168").abs()
    volume_z = _column(frame, "volume_zscore_168")
    eps = 1e-8

    result = pd.DataFrame(index=frame.index)
    result["state_return_z_24"] = r1 / (vol24 + eps)
    result["state_trend_strength_24"] = r24.abs() / (vol168 * np.sqrt(24.0) + eps)
    result["state_volatility_ratio"] = vol24 / (vol168 + eps)
    result["state_volume_shock"] = volume_z.clip(-10, 10)
    result["state_return_24"] = r24
    result["state_volatility_24"] = vol24
    result["state_volatility_168"] = vol168
    result["state_shock_score"] = pd.concat(
        [result["state_return_z_24"].abs(), result["state_volume_shock"].abs(), result["state_volatility_ratio"]], axis=1
    ).max(axis=1)
    result["state_is_shock"] = (result["state_shock_score"] >= 3.0).astype(float)
    result["state_is_high_volatility"] = (result["state_volatility_ratio"] >= 1.5).astype(float)
    result["state_is_trend"] = (result["state_trend_strength_24"] >= 1.0).astype(float)
    return result.replace([np.inf, -np.inf], np.nan)
