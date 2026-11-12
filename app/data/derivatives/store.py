"""Storage of derivatives history: data/derivatives/<dataset>/<SYMBOL>.parquet."""

import os
from pathlib import Path

import pandas as pd

DATASETS = ("funding", "premium", "metrics", "dvol")


class DerivativesStore:
    def __init__(self, data_dir: Path | str) -> None:
        self.root = Path(data_dir) / "derivatives"

    def path(self, dataset: str, symbol: str) -> Path:
        return self.root / dataset / f"{symbol.upper()}.parquet"

    def load(self, dataset: str, symbol: str) -> pd.DataFrame | None:
        path = self.path(dataset, symbol)
        return pd.read_parquet(path) if path.exists() else None

    def load_all(self, symbol: str) -> dict[str, pd.DataFrame]:
        return {name: df for name in DATASETS if (df := self.load(name, symbol)) is not None}

    def append(self, dataset: str, symbol: str, new_rows: pd.DataFrame) -> pd.DataFrame:
        """Merge new rows (deduplicated by available_at) and write atomically."""
        existing = self.load(dataset, symbol)
        frame = new_rows if existing is None else pd.concat([existing, new_rows], ignore_index=True)
        frame = frame.drop_duplicates("available_at", keep="last").sort_values("available_at").reset_index(drop=True)
        path = self.path(dataset, symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return frame

    def last_available(self, dataset: str, symbol: str) -> pd.Timestamp | None:
        frame = self.load(dataset, symbol)
        if frame is None or frame.empty:
            return None
        return frame["available_at"].max()
