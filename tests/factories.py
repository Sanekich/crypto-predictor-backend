"""Synthetic market data for tests (not real prices)."""

from typing import Any

import numpy as np
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


def make_random_ohlcv(count: int, seed: int = 0, start_ms: int = START_MS, step_ms: int = HOUR_MS) -> pd.DataFrame:
    """Random-walk OHLCV with realistic structure (for feature tests)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, count)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.005, count)) * close
    volume = rng.gamma(2.0, 50.0, count)
    volume[rng.random(count) < 0.01] = 0.0  # occasional zero-volume candle
    timestamps = pd.to_datetime(start_ms + np.arange(count) * step_ms, unit="ms", utc=True)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(10, 1000, count),
            "taker_buy_base": volume * rng.uniform(0.3, 0.7, count),
            "taker_buy_quote": volume * close * 0.5,
            "close_time": timestamps + pd.Timedelta(milliseconds=step_ms - 1),
        }
    )


def make_ohlcv(count: int, start_ms: int = START_MS, step_ms: int = HOUR_MS) -> pd.DataFrame:
    return normalize_klines(klines_to_raw_frame(make_klines(count, start_ms, step_ms)))
