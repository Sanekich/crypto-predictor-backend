"""Hourly Parquet files for live data.

Layout: data/live/<dataset>/<SYMBOL>/<YYYY-MM-DD>/<HH>.parquet (UTC).

Rows are buffered and flushed periodically. A flush merges new rows into the
existing hour file (deduplicated by the dataset key), so restarts never lose or
duplicate data that was already written.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

Row = dict[str, Any]

# Columns that identify a row. Per-minute datasets use the timestamp alone;
# event datasets can have several events in the same millisecond.
DATASET_KEYS: dict[str, list[str]] = {
    "connection_events": ["timestamp", "connection", "event"],
    "liquidations": ["timestamp", "side", "price", "qty"],
}


def dataset_key(dataset: str) -> list[str]:
    return DATASET_KEYS.get(dataset, ["timestamp"])


class LiveDataWriter:
    def __init__(self, live_dir: Path | str, symbol: str) -> None:
        self.live_dir = Path(live_dir)
        self.symbol = symbol.upper()
        self._pending: dict[str, list[Row]] = defaultdict(list)
        self.rows_written: dict[str, int] = defaultdict(int)

    def add(self, dataset: str, rows: list[Row]) -> None:
        if rows:
            self._pending[dataset].extend(rows)

    def pending_count(self, dataset: str) -> int:
        return len(self._pending.get(dataset, []))

    def hour_path(self, dataset: str, timestamp: pd.Timestamp) -> Path:
        return self.live_dir / dataset / self.symbol / timestamp.strftime("%Y-%m-%d") / f"{timestamp:%H}.parquet"

    def flush(self) -> int:
        """Write all buffered rows. Returns the number of rows flushed."""
        flushed = 0
        pending, self._pending = self._pending, defaultdict(list)
        for dataset, rows in pending.items():
            frame = pd.DataFrame(rows)
            hours = frame["timestamp"].dt.floor("h")
            for hour, part in frame.groupby(hours):
                self._merge_write(self.hour_path(dataset, hour), part, dataset_key(dataset))
            self.rows_written[dataset] += len(frame)
            flushed += len(frame)
        return flushed

    @staticmethod
    def _merge_write(path: Path, new_rows: pd.DataFrame, key: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined = (
            combined.drop_duplicates(key, keep="last")
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )
        tmp = path.with_name(path.name + ".tmp")
        combined.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def write_status(self, status: dict[str, Any]) -> Path:
        path = self.live_dir / f"_status_{self.symbol}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return path


def load_live(live_dir: Path | str, dataset: str, symbol: str) -> pd.DataFrame:
    """Read all hourly files of a dataset into one frame sorted by time."""
    root = Path(live_dir) / dataset / symbol.upper()
    files = sorted(root.glob("*/*.parquet"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return (
        frame.drop_duplicates(dataset_key(dataset), keep="last")
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
