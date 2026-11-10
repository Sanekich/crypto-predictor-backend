"""Candle intervals (Binance notation) and time conversion helpers."""

from datetime import datetime, timezone

import pandas as pd

_MINUTE = 60_000
_HOUR = 60 * _MINUTE
_DAY = 24 * _HOUR

INTERVAL_MS: dict[str, int] = {
    "1m": _MINUTE,
    "3m": 3 * _MINUTE,
    "5m": 5 * _MINUTE,
    "15m": 15 * _MINUTE,
    "30m": 30 * _MINUTE,
    "1h": _HOUR,
    "2h": 2 * _HOUR,
    "4h": 4 * _HOUR,
    "6h": 6 * _HOUR,
    "8h": 8 * _HOUR,
    "12h": 12 * _HOUR,
    "1d": _DAY,
    "3d": 3 * _DAY,
    "1w": 7 * _DAY,
}

# Intervals whose candles start on a fixed grid counted from the Unix epoch.
# Weekly candles start on Monday (the epoch was a Thursday), so they are excluded.
GRID_ALIGNED_INTERVALS = frozenset(INTERVAL_MS) - {"3d", "1w"}

UNIX_EPOCH = pd.Timestamp(0, tz="UTC")


def interval_to_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        supported = ", ".join(INTERVAL_MS)
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported: {supported}"
        ) from None


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    return pd.Timedelta(milliseconds=interval_to_ms(interval))


def datetime_to_ms(value: datetime) -> int:
    """Convert a timezone-aware datetime to Unix milliseconds."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    return int(value.timestamp() * 1000)


def ms_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
