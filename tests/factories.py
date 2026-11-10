"""Synthetic market data for tests (not real prices)."""

from typing import Any

import pandas as pd

from app.data.processing.cleaning import klines_to_raw_frame, normalize_klines

HOUR_MS = 3_600_000
START_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z


def make_klines(count: int, start_ms: int = START_MS, step_ms: int = HOUR_MS) -> list[list[Any]]:
    """Klines in the exact Binance response format."""
    rows = []
    for i in range(count):
        open_time = start_ms + i * step_ms
        price = 100.0 + i
        rows.append(
            [
                open_time,
                f"{price:.8f}",
                f"{price + 2:.8f}",
                f"{price - 1:.8f}",
                f"{price + 1:.8f}",
                "10.50000000",
                open_time + step_ms - 1,
                "1050.00000000",
                42,
                "5.00000000",
                "500.00000000",
                "0",
            ]
        )
    return rows


def make_ohlcv(count: int, start_ms: int = START_MS, step_ms: int = HOUR_MS) -> pd.DataFrame:
    return normalize_klines(klines_to_raw_frame(make_klines(count, start_ms, step_ms)))
