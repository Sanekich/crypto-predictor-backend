"""File storage for market data: raw and processed Parquet files plus quality reports.

Layout:
    data/raw/<provider>/<SYMBOL>_<interval>.parquet
    data/processed/<provider>/<SYMBOL>_<interval>.parquet
    data/processed/<provider>/<SYMBOL>_<interval>.report.json
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


class MarketDataStorage:
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)

    def _path(self, layer: str, provider: str, symbol: str, interval: str, suffix: str) -> Path:
        return self.data_dir / layer / provider / f"{symbol.upper()}_{interval}{suffix}"

    def raw_path(self, provider: str, symbol: str, interval: str) -> Path:
        return self._path("raw", provider, symbol, interval, ".parquet")

    def processed_path(self, provider: str, symbol: str, interval: str) -> Path:
        return self._path("processed", provider, symbol, interval, ".parquet")

    def report_path(self, provider: str, symbol: str, interval: str) -> Path:
        return self._path("processed", provider, symbol, interval, ".report.json")

    def load_raw(self, provider: str, symbol: str, interval: str) -> pd.DataFrame | None:
        return self._read_parquet(self.raw_path(provider, symbol, interval))

    def load_processed(self, provider: str, symbol: str, interval: str) -> pd.DataFrame | None:
        return self._read_parquet(self.processed_path(provider, symbol, interval))

    def save_raw(self, df: pd.DataFrame, provider: str, symbol: str, interval: str) -> Path:
        path = self.raw_path(provider, symbol, interval)
        _atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))
        return path

    def save_processed(self, df: pd.DataFrame, provider: str, symbol: str, interval: str) -> Path:
        path = self.processed_path(provider, symbol, interval)
        _atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))
        return path

    def save_report(self, report: dict[str, Any], provider: str, symbol: str, interval: str) -> Path:
        path = self.report_path(provider, symbol, interval)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        _atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
        return path

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame | None:
        return pd.read_parquet(path) if path.exists() else None


def _atomic_write(path: Path, writer: Callable[[Path], object]) -> None:
    """Write to a temp file first so an interrupted run never leaves a broken file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)
