"""Download derivatives history from public sources.

Every frame has an `available_at` column (UTC): the moment the value was
publicly known. Features may only use rows with available_at <= the close of
the candle they are computed for - this is what prevents look-ahead leakage.

| Dataset        | Source                                   | Step   | available_at              |
|----------------|------------------------------------------|--------|---------------------------|
| funding        | Binance /fapi/v1/fundingRate             | 8 h    | fundingTime               |
| premium        | Binance /fapi/v1/premiumIndexKlines (1h) | 1 h    | candle open + 1 h         |
| metrics        | data.binance.vision daily metrics zips   | 5 min  | create_time               |
| dvol           | Deribit get_volatility_index_data (1h)   | 1 h    | candle open + 1 h         |
"""

import io
import logging
import time
import zipfile
from collections.abc import Callable
from datetime import date, timedelta

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
VISION = "https://data.binance.vision/data/futures/um/daily/metrics"
DERIBIT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
HOUR_MS = 3_600_000


def _get(client: httpx.Client, url: str, params: dict | None = None, retries: int = 4,
         sleep: Callable[[float], None] = time.sleep) -> httpx.Response:
    for attempt in range(retries + 1):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            if attempt == retries:
                raise
            sleep(2**attempt)
            continue
        if response.status_code in (429, 418) or response.status_code >= 500:
            if attempt == retries:
                response.raise_for_status()
            sleep(int(response.headers.get("Retry-After", 2**attempt)))
            continue
        return response
    raise RuntimeError("unreachable")  # pragma: no cover


def fetch_funding(client: httpx.Client, symbol: str, start_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = start_ms
    while True:
        batch = _get(client, f"{FAPI}/fapi/v1/fundingRate", {"symbol": symbol, "startTime": cursor, "limit": 1000}).json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["available_at", "funding_rate"])
    df = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "available_at": pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True),
            "funding_rate": df["fundingRate"].astype(float),
        }
    )


def fetch_premium(client: httpx.Client, symbol: str, start_ms: int) -> pd.DataFrame:
    """Premium index = (perpetual price - spot index) / index: the futures basis."""
    rows: list[list] = []
    cursor = start_ms
    while True:
        batch = _get(
            client,
            f"{FAPI}/fapi/v1/premiumIndexKlines",
            {"symbol": symbol, "interval": "1h", "startTime": cursor, "limit": 1500},
        ).json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + HOUR_MS
        if len(batch) < 1500:
            break
    if not rows:
        return pd.DataFrame(columns=["timestamp", "available_at", "premium_close", "premium_high", "premium_low"])
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["open_time", "open", "high", "low", "close"]
    ts = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "available_at": ts + pd.Timedelta(hours=1),
            "premium_close": df["close"].astype(float),
            "premium_high": df["high"].astype(float),
            "premium_low": df["low"].astype(float),
        }
    )


METRIC_COLUMNS = {
    "sum_open_interest": "open_interest",
    "sum_open_interest_value": "open_interest_value",
    "count_toptrader_long_short_ratio": "top_trader_account_ratio",
    "sum_toptrader_long_short_ratio": "top_trader_position_ratio",
    "count_long_short_ratio": "global_account_ratio",
    "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
}


def fetch_metrics_day(client: httpx.Client, symbol: str, day: date) -> pd.DataFrame | None:
    """One day of 5-minute futures metrics; None if the archive has no file for that day."""
    url = f"{VISION}/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
    response = _get(client, url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            raw = pd.read_csv(handle)
    frame = pd.DataFrame({"available_at": pd.to_datetime(raw["create_time"], utc=True)})
    for source, target in METRIC_COLUMNS.items():
        frame[target] = pd.to_numeric(raw.get(source), errors="coerce")
    return frame


def fetch_metrics(client: httpx.Client, symbol: str, start: date, end: date,
                  progress: Callable[[date], None] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    day = start
    while day <= end:
        frame = fetch_metrics_day(client, symbol, day)
        if frame is not None and not frame.empty:
            frames.append(frame)
        if progress and day.day == 1:
            progress(day)
        day += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["available_at", *METRIC_COLUMNS.values()])
    return pd.concat(frames, ignore_index=True)


def fetch_dvol(client: httpx.Client, currency: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Deribit implied volatility index (annualised %, 30-day), hourly candles."""
    rows: list[list] = []
    chunk = 1000 * HOUR_MS
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + chunk, end_ms)
        result = _get(
            client,
            DERIBIT,
            {"currency": currency, "start_timestamp": cursor, "end_timestamp": chunk_end, "resolution": 3600},
        ).json()["result"]
        rows.extend(result.get("data", []))
        cursor = chunk_end + HOUR_MS
    if not rows:
        return pd.DataFrame(columns=["timestamp", "available_at", "dvol"])
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"]).drop_duplicates("t")
    ts = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    return pd.DataFrame({"timestamp": ts, "available_at": ts + pd.Timedelta(hours=1), "dvol": df["close"].astype(float)})
