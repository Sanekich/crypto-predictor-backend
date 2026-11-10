from datetime import datetime, timezone

import pandas as pd
import pytest

from app.data.processing.cleaning import (
    OHLCV_COLUMNS,
    clean_ohlcv,
    klines_to_raw_frame,
    normalize_klines,
)
from tests.factories import make_klines, make_ohlcv


def test_normalize_klines_types() -> None:
    df = normalize_klines(klines_to_raw_frame(make_klines(3)))
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert df["close"].dtype == "float64"
    assert df["close"].iloc[1] == pytest.approx(102.0)


def test_raw_frame_keeps_source_strings() -> None:
    raw = klines_to_raw_frame(make_klines(1))
    assert raw["open"].iloc[0] == "100.00000000"
    assert "ignore" not in raw.columns


def test_clean_drops_unclosed_candle() -> None:
    df = make_ohlcv(5)
    as_of = datetime(2024, 1, 1, 4, 30, tzinfo=timezone.utc)  # candle 04:00 is still open
    cleaned = clean_ohlcv(df, as_of)
    assert len(cleaned) == 4
    assert cleaned["timestamp"].max() == pd.Timestamp("2024-01-01T03:00:00Z")


def test_clean_sorts_and_keeps_last_duplicate() -> None:
    df = make_ohlcv(3)
    updated = df.iloc[[1]].copy()
    updated["close"] = 999.0
    messy = pd.concat([df.iloc[[2, 1, 0]], updated], ignore_index=True)

    cleaned = clean_ohlcv(messy, datetime(2030, 1, 1, tzinfo=timezone.utc))

    assert cleaned["timestamp"].is_monotonic_increasing
    assert len(cleaned) == 3
    assert cleaned["close"].iloc[1] == 999.0


def test_clean_requires_timezone() -> None:
    with pytest.raises(ValueError):
        clean_ohlcv(make_ohlcv(2), datetime(2024, 1, 1))
