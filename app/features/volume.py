"""Volume features. log1p is used because volume can be zero."""

import numpy as np
import pandas as pd


def volume_change(volume: pd.Series, periods: int = 1) -> pd.Series:
    log_volume = np.log1p(volume)
    return log_volume - log_volume.shift(periods)


def volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    """Current volume relative to its rolling mean."""
    mean = volume.rolling(window, min_periods=window).mean()
    return volume / mean.where(mean > 0)


def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    log_volume = np.log1p(volume)
    rolling = log_volume.rolling(window, min_periods=window)
    std = rolling.std()
    return (log_volume - rolling.mean()) / std.where(std > 0)


def taker_buy_ratio(taker_buy_base: pd.Series, volume: pd.Series) -> pd.Series:
    """Share of volume bought by market (taker) orders, 0..1."""
    return taker_buy_base / volume.where(volume > 0)
