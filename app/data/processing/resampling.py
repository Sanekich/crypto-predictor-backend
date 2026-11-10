"""Alignment of OHLCV data to a regular time grid."""

import pandas as pd

from app.data.intervals import interval_to_timedelta


def reindex_to_grid(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Insert empty (NaN) rows for missing candles so that one row == one interval.

    Row-based operations (shift, rolling) then correspond to exact time
    offsets, and any window that touches a gap yields NaN instead of
    silently spanning a longer period.
    """
    if df.empty:
        return df.copy()

    step = interval_to_timedelta(interval)
    ts = df["timestamp"]
    grid = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=step).as_unit(ts.dt.unit)
    out = df.set_index("timestamp").reindex(grid)
    out.index.name = "timestamp"
    return out.reset_index()
