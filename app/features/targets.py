"""Prediction targets. These intentionally look into the future and must
never be used as model inputs."""

import re

import pandas as pd

from app.data.intervals import interval_to_ms

_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
_DURATION_RE = re.compile(r"^(\d+)([mhdw])$")


def horizon_to_periods(horizon: str, interval: str) -> int:
    """Number of candles in the horizon, e.g. ('24h', '1h') -> 24, ('1d', '4h') -> 6."""
    match = _DURATION_RE.match(horizon)
    if not match:
        raise ValueError(f"Invalid horizon '{horizon}', expected e.g. 24h, 3d, 1w")
    horizon_ms = int(match.group(1)) * _UNIT_MS[match.group(2)]
    step_ms = interval_to_ms(interval)
    if horizon_ms <= 0 or horizon_ms % step_ms:
        raise ValueError(f"Horizon '{horizon}' is not a positive multiple of interval '{interval}'")
    return horizon_ms // step_ms


def future_return(close: pd.Series, periods: int) -> pd.Series:
    """close[t + periods] / close[t] - 1. NaN when the future candle is unknown."""
    return close.shift(-periods) / close - 1


def future_direction(future_ret: pd.Series) -> pd.Series:
    """1.0 if the future return is positive, 0.0 otherwise, NaN if unknown."""
    return (future_ret > 0).astype("float64").where(future_ret.notna())


def future_direction_with_neutral(future_ret: pd.Series, neutral_return: float = 0.001) -> pd.Series:
    """Return -1/0/1 for down/neutral/up moves beyond a minimum return.

    This is intentionally opt-in: existing binary direction models continue to use
    :func:`future_direction` until a dedicated multiclass trainer is introduced.
    """
    if neutral_return < 0:
        raise ValueError("neutral_return must be non-negative")
    result = pd.Series(0.0, index=future_ret.index)
    result = result.mask(future_ret > neutral_return, 1.0)
    result = result.mask(future_ret < -neutral_return, -1.0)
    return result.where(future_ret.notna())
