"""Symmetric barrier outcomes: "which comes first, a rise of X% or a fall of X%?".

For a candle that closes at price P the up barrier is P * (1 + threshold) and the
down barrier P * (1 - threshold). Looking at the following candles, at most
`max_periods` of them:

- "up"       a later candle's high reaches the up barrier before any low reaches the down barrier
- "down"     the opposite
- "both"     both barriers are touched inside the same candle, so the order is unknown
- "none"     neither barrier is reached within max_periods (no significant move)
- "unknown"  the future is not fully known yet (end of data) or the candle is missing

Why symmetric: with equal thresholds a 75% hit rate means the direction was really
right 3 times out of 4. An asymmetric target ("+0.5% before -3%") would show a high
hit rate while the few misses cost more than all hits earn.

These labels look into the future and must never be used as model inputs.
"""

import numpy as np
import pandas as pd

OUTCOMES = ("up", "down", "both", "none", "unknown")
RESOLVED = ("up", "down")

_NOT_HIT = np.iinfo(np.int64).max


def barrier_outcomes(grid: pd.DataFrame, threshold: float, max_periods: int) -> pd.DataFrame:
    """Outcome per row of `grid` (a regular candle grid with high, low, close).

    Returns a frame with the same index and columns:
    - outcome: one of OUTCOMES;
    - label: 1.0 for "up", 0.0 for "down", NaN otherwise (training target);
    - resolve_periods: candles until the first barrier hit; max_periods when nothing was hit.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be a fraction in (0, 1), e.g. 0.01 for 1%")
    if max_periods < 1:
        raise ValueError("max_periods must be at least 1")

    close = grid["close"].to_numpy(dtype="float64")
    high = grid["high"].to_numpy(dtype="float64")
    low = grid["low"].to_numpy(dtype="float64")
    n = len(close)
    up_level = close * (1 + threshold)
    down_level = close * (1 - threshold)

    first_up = np.full(n, _NOT_HIT, dtype=np.int64)
    first_down = np.full(n, _NOT_HIT, dtype=np.int64)
    for k in range(1, min(max_periods, n - 1) + 1):
        future_high = np.full(n, np.nan)
        future_low = np.full(n, np.nan)
        future_high[: n - k] = high[k:]
        future_low[: n - k] = low[k:]
        with np.errstate(invalid="ignore"):  # NaN (missing candle) never counts as a hit
            hit_up = (first_up == _NOT_HIT) & (future_high >= up_level)
            hit_down = (first_down == _NOT_HIT) & (future_low <= down_level)
        first_up[hit_up] = k
        first_down[hit_down] = k

    first_hit = np.minimum(first_up, first_down)
    resolved = first_hit != _NOT_HIT
    known_steps = np.minimum(max_periods, n - 1 - np.arange(n))  # candles of known future

    outcome = np.full(n, "none", dtype=object)
    outcome[resolved & (first_up < first_down)] = "up"
    outcome[resolved & (first_down < first_up)] = "down"
    outcome[resolved & (first_up == first_down)] = "both"
    outcome[~resolved & (known_steps < max_periods)] = "unknown"
    outcome[np.isnan(close)] = "unknown"

    label = np.where(outcome == "up", 1.0, np.where(outcome == "down", 0.0, np.nan))
    resolve_periods = np.where(resolved, first_hit, max_periods).astype(int)
    return pd.DataFrame({"outcome": outcome, "label": label, "resolve_periods": resolve_periods}, index=grid.index)
