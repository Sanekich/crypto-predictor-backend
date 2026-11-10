"""Quality checks for OHLCV datasets (see architecture doc, section 5.3)."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.data.intervals import GRID_ALIGNED_INTERVALS, UNIX_EPOCH, interval_to_timedelta

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class Gap:
    """Missing candles between two consecutive timestamps."""

    after: pd.Timestamp
    before: pd.Timestamp
    missing_candles: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "after": self.after.isoformat(),
            "before": self.before.isoformat(),
            "missing_candles": self.missing_candles,
        }


@dataclass
class ValidationReport:
    interval: str
    rows: int = 0
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    errors: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Errors make the dataset unusable. Gaps are documented but allowed."""
        return not self.errors

    @property
    def missing_candles(self) -> int:
        return sum(gap.missing_candles for gap in self.gaps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "rows": self.rows,
            "first_timestamp": self.first_timestamp.isoformat() if self.first_timestamp else None,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "gap_count": len(self.gaps),
            "missing_candles": self.missing_candles,
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


def validate_ohlcv(
    df: pd.DataFrame,
    interval: str,
    as_of: datetime | None = None,
) -> ValidationReport:
    """Check an OHLCV frame of a single symbol and interval.

    `as_of` enables the check that no unclosed candle is present.
    """
    step = interval_to_timedelta(interval)
    report = ValidationReport(interval=interval, rows=len(df))

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        report.errors.append(f"missing columns: {', '.join(missing_columns)}")
        return report

    if df.empty:
        report.errors.append("dataset is empty")
        return report

    ts = df["timestamp"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype) or str(ts.dt.tz) != "UTC":
        report.errors.append(f"timestamp must be timezone-aware UTC, got {ts.dtype}")
        return report

    for col, count in df[REQUIRED_COLUMNS].isna().sum().items():
        if count:
            report.errors.append(f"missing values in '{col}': {int(count)} row(s)")
    if ts.isna().any():
        return report

    report.first_timestamp = ts.min()
    report.last_timestamp = ts.max()

    def check(mask: pd.Series, message: str) -> None:
        count = int(mask.sum())
        if count:
            first = ts[mask].iloc[0].isoformat()
            report.errors.append(f"{message}: {count} row(s), first at {first}")

    check(ts.duplicated(), "duplicate timestamps")
    check(ts.diff() < pd.Timedelta(0), "timestamps out of order")
    if interval in GRID_ALIGNED_INTERVALS:
        check((ts - UNIX_EPOCH) % step != pd.Timedelta(0), f"timestamps not aligned to {interval} grid")

    check(pd.Series(np.isinf(df[PRICE_COLUMNS + ["volume"]]).any(axis=1)), "infinite values")
    check((df[PRICE_COLUMNS] <= 0).any(axis=1), "non-positive prices")
    check(df["high"] < df[["open", "close", "low"]].max(axis=1), "high below open/close/low")
    check(df["low"] > df[["open", "close"]].min(axis=1), "low above open/close")
    check(df["volume"] < 0, "negative volume")

    if as_of is not None and "close_time" in df.columns:
        check(df["close_time"] >= pd.Timestamp(as_of), "unclosed candles")

    if ts.is_monotonic_increasing:
        diffs = ts.diff()
        for pos in np.flatnonzero((diffs > step).to_numpy()):
            after, before = ts.iloc[pos - 1], ts.iloc[pos]
            missing = int((before - after) / step) - 1
            report.gaps.append(Gap(after=after, before=before, missing_candles=missing))

    return report
