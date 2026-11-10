"""Chronological train / validation / test split with purging.

A target at time t covers the period (t, t + horizon]. If the last training
rows were right before the validation period, their targets would overlap
with validation prices (label leakage). A gap of `horizon` is therefore
removed before the validation and before the test period.

    |------ train ------| gap |--- validation ---| gap |--- test ---|
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    gap: pd.Timedelta

    def items(self) -> list[tuple[str, np.ndarray]]:
        return [("train", self.train), ("validation", self.validation), ("test", self.test)]

    def describe(self, timestamps: pd.Series) -> dict[str, Any]:
        def period(idx: np.ndarray) -> dict[str, Any]:
            ts = timestamps.iloc[idx]
            return {"rows": len(idx), "start": ts.iloc[0].isoformat(), "end": ts.iloc[-1].isoformat()}

        return {
            "gap": str(self.gap),
            **{name: period(idx) for name, idx in self.items()},
        }


def chronological_split(
    timestamps: pd.Series,
    gap: pd.Timedelta,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> TimeSplit:
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("fractions must be between 0 and 1")
    if validation_fraction + test_fraction >= 0.9:
        raise ValueError("validation + test fractions leave too little training data")

    ts = timestamps.reset_index(drop=True)
    n = len(ts)
    validation_start = ts.iloc[int(n * (1 - validation_fraction - test_fraction))]
    test_start = ts.iloc[int(n * (1 - test_fraction))]

    positions = np.arange(n)
    train = positions[(ts < validation_start - gap).to_numpy()]
    validation = positions[((ts >= validation_start) & (ts < test_start - gap)).to_numpy()]
    test = positions[(ts >= test_start).to_numpy()]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("not enough data for a train/validation/test split with this gap")

    return TimeSplit(train, validation, test, validation_start, test_start, gap)
