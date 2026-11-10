"""Volatility features."""

import pandas as pd

from app.features.returns import log_return


def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Standard deviation of 1-period log returns over the window."""
    return log_return(close, 1).rolling(window, min_periods=window).std()


def candle_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """(high - low) / close: intra-candle range relative to price."""
    return (high - low) / close
