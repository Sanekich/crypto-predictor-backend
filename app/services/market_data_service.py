"""Access to processed market data with on-demand incremental refresh."""

import logging
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.core.exceptions import DataNotAvailableError
from app.data.collection.binance_client import BinanceAPIError
from app.data.collection.collector import KlinesSource, MarketDataCollector
from app.data.intervals import interval_to_timedelta
from app.data.storage import MarketDataStorage

logger = logging.getLogger(__name__)

SourceFactory = Callable[[], AbstractContextManager[KlinesSource]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketDataService:
    """Returns the processed OHLCV dataset, downloading new closed candles when
    a newer one should already exist. Refresh attempts are throttled so that
    many API requests do not turn into many exchange requests."""

    def __init__(
        self,
        storage: MarketDataStorage,
        source_factory: SourceFactory,
        provider: str = "binance",
        clock: Callable[[], datetime] = _utc_now,
        min_refresh_interval: timedelta = timedelta(seconds=60),
    ) -> None:
        self._storage = storage
        self._source_factory = source_factory
        self._provider = provider
        self._clock = clock
        self._min_refresh_interval = min_refresh_interval
        self._lock = threading.Lock()
        self._last_attempt: dict[tuple[str, str], datetime] = {}
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    @staticmethod
    def has_newer_closed_candle(df: pd.DataFrame, interval: str, now: datetime) -> bool:
        """True if a candle after the last stored one has already closed."""
        last_open = df["timestamp"].iloc[-1]
        return last_open + 2 * interval_to_timedelta(interval) <= pd.Timestamp(now)

    def get_ohlcv(self, symbol: str, interval: str, refresh: bool = True) -> pd.DataFrame:
        key = (symbol.upper(), interval)
        with self._lock:
            df = self._cache.get(key)
            if df is None:
                df = self._storage.load_processed(self._provider, *key)
            if df is None or df.empty:
                raise DataNotAvailableError(
                    f"No market data for {key[0]} {interval}. Run scripts/download_data.py first."
                )

            now = self._clock()
            last_attempt = self._last_attempt.get(key)
            throttled = last_attempt is not None and now - last_attempt < self._min_refresh_interval
            if refresh and not throttled and self.has_newer_closed_candle(df, interval, now):
                self._last_attempt[key] = now
                df = self._refresh(key, df)

            self._cache[key] = df
            return df

    def _refresh(self, key: tuple[str, str], current: pd.DataFrame) -> pd.DataFrame:
        symbol, interval = key
        try:
            with self._source_factory() as source:
                result = MarketDataCollector(source, self._storage, self._provider, self._clock).collect(
                    symbol, interval, start=current["timestamp"].iloc[0].to_pydatetime()
                )
        except (BinanceAPIError, OSError) as exc:
            logger.warning("Market data refresh failed for %s %s: %s", symbol, interval, exc)
            return current

        if not result.report.is_valid:
            logger.error("Refreshed data failed validation, keeping previous dataset: %s", result.report.errors)
            return current

        refreshed = self._storage.load_processed(self._provider, symbol, interval)
        return refreshed if refreshed is not None else current
