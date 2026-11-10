"""Price return features. All functions use only past and current values."""

import numpy as np
import pandas as pd


def simple_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """close[t] / close[t - periods] - 1"""
    return close / close.shift(periods) - 1


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """ln(close[t] / close[t - periods])"""
    return np.log(close / close.shift(periods))
