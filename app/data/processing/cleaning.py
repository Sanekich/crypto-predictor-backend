"""Conversion of raw Binance klines into a clean OHLCV DataFrame."""

from datetime import datetime
from typing import Any

import pandas as pd

RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
]

OHLCV_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "close_time",
]


def klines_to_raw_frame(rows: list[list[Any]]) -> pd.DataFrame:
    """Raw klines as received from the API. Prices stay as strings (exact source values)."""
    df = pd.DataFrame(rows, columns=RAW_COLUMNS).drop(columns="ignore")
    return df.astype(
        {
            "open_time": "int64",
            "close_time": "int64",
            "trades": "int64",
            **{col: "string" for col in FLOAT_COLUMNS},
        }
    )


def normalize_klines(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed OHLCV frame: UTC timestamps (candle open time) and float prices."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["open_time"], unit="ms", utc=True),
            **{
                col: pd.to_numeric(raw[col].astype(object), errors="coerce").astype("float64")
                for col in FLOAT_COLUMNS
            },
            "trades": raw["trades"].astype("int64"),
            "close_time": pd.to_datetime(raw["close_time"], unit="ms", utc=True),
        }
    )
    return df[OHLCV_COLUMNS]


def clean_ohlcv(df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Drop unclosed candles, sort by time and remove duplicate timestamps.

    Values are not modified or filled: invalid rows and gaps are reported
    by validation instead of being silently fixed.
    """
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        raise ValueError("as_of must be timezone-aware (UTC)")

    closed = df[df["close_time"] < as_of_ts]
    return (
        closed.sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
