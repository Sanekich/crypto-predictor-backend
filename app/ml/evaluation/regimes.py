"""Simple, causal market-regime labels for evaluation and filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIMES = ("trend_up", "trend_down", "range", "high_volatility")


def regime_features(close: pd.Series, trend_window: int = 168, volatility_window: int = 24) -> pd.DataFrame:
    """Return causal trend and volatility features using only data through t."""
    log_returns = np.log(close / close.shift(1))
    return pd.DataFrame(
        {
            "trend_return": close / close.shift(trend_window) - 1,
            "volatility": log_returns.rolling(volatility_window, min_periods=volatility_window).std(),
        },
        index=close.index,
    )


def label_regimes(features: pd.DataFrame, calibration_mask: pd.Series) -> pd.Series:
    """Label regimes using thresholds fitted only on calibration rows."""
    calibration = features.loc[calibration_mask].dropna()
    if calibration.empty:
        raise ValueError("no calibration rows available for regime thresholds")
    trend_low, trend_high = calibration["trend_return"].quantile([0.25, 0.75])
    vol_high = calibration["volatility"].quantile(0.75)

    result = pd.Series("range", index=features.index, dtype="string")
    result.loc[features["trend_return"] >= trend_high] = "trend_up"
    result.loc[features["trend_return"] <= trend_low] = "trend_down"
    result.loc[features["volatility"] >= vol_high] = "high_volatility"
    return result.where(features.notna().all(axis=1))
