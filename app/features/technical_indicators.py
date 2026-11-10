"""Technical indicators. All are causal: the value at t depends only on rows <= t.

Windows are given in candles. A window is only computed once it is full
(min_periods == window), so early rows and rows right after a gap are NaN.
"""

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing (used by RSI and ATR)."""
    return series.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index in the range 0..100 (Wilder)."""
    delta = close.diff()
    avg_gain = _wilder(delta.clip(lower=0), window)
    avg_loss = _wilder(-delta.clip(upper=0), window)

    value = 100 - 100 / (1 + avg_gain / avg_loss)
    value = value.where(avg_loss != 0, 100.0)
    value = value.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return value.where(avg_gain.notna() & avg_loss.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram) in price units."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return line, signal_line, line - signal_line


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    value = ranges.max(axis=1)
    return value.where(high.notna() & low.notna())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder) in price units."""
    return _wilder(true_range(high, low, close), window)
