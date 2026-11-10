"""Journal of issued high-confidence signals and their real outcomes.

Every signal is stored once per (horizon, candle). Outcomes are not stored:
they are computed from market data when the journal is read, so a signal
becomes "resolved" automatically once its target time has passed.
"""

import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.intervals import interval_to_timedelta

KEY = ["horizon", "signal_time"]


class SignalJournal:
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def _path(self, symbol: str) -> Path:
        return self.directory / f"{symbol.upper()}.parquet"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._path(symbol)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def record(self, symbol: str, entry: dict[str, Any]) -> None:
        with self._lock:
            existing = self.load(symbol)
            combined = pd.concat([existing, pd.DataFrame([entry])], ignore_index=True)
            combined = combined.drop_duplicates(KEY, keep="first").sort_values("signal_time").reset_index(drop=True)
            path = self._path(symbol)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            combined.to_parquet(tmp, index=False)
            os.replace(tmp, path)

    def with_outcomes(self, symbol: str, ohlcv: pd.DataFrame, interval: str, limit: int = 50) -> pd.DataFrame:
        """Recent signals with outcome price and correctness (None while pending)."""
        journal = self.load(symbol)
        if journal.empty:
            return journal
        journal = journal.sort_values("signal_time", ascending=False).head(limit).copy()

        # Close price at time T is the close of the candle that opened at T - interval
        step = interval_to_timedelta(interval)
        closes = ohlcv.set_index("timestamp")["close"]
        outcome_open = journal["target_time"] - step
        journal["outcome_price"] = outcome_open.map(closes)
        resolved = journal["outcome_price"].notna()
        went_up = journal["outcome_price"] > journal["price_at_signal"]
        journal["correct"] = (went_up == (journal["direction"] == "up")).where(resolved)
        return journal
