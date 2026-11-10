"""Market data pipeline: fetch -> save raw -> clean -> validate -> save processed."""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from app.data.intervals import datetime_to_ms, interval_to_ms, ms_to_datetime
from app.data.processing.cleaning import clean_ohlcv, klines_to_raw_frame, normalize_klines
from app.data.storage import MarketDataStorage
from app.data.validation.market_data_validator import ValidationReport, validate_ohlcv

logger = logging.getLogger(__name__)


class KlinesSource(Protocol):
    def iter_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int | None = None
    ) -> Iterator[list[list[Any]]]: ...


@dataclass
class CollectionResult:
    symbol: str
    interval: str
    fetched_rows: int
    raw_rows: int
    processed_rows: int
    report: ValidationReport
    raw_path: Path
    report_path: Path
    processed_path: Path | None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketDataCollector:
    def __init__(
        self,
        source: KlinesSource,
        storage: MarketDataStorage,
        provider: str = "binance",
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._source = source
        self._storage = storage
        self._provider = provider
        self._clock = clock

    def collect(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        full_refresh: bool = False,
    ) -> CollectionResult:
        """Download candles and rebuild the processed dataset.

        If raw data already exists, only new candles are fetched (starting from
        the last stored candle, which may have been unclosed at the previous run).
        """
        symbol = symbol.upper()
        interval_to_ms(interval)
        key = (self._provider, symbol, interval)

        existing = None if full_refresh else self._storage.load_raw(*key)
        if existing is not None and not existing.empty:
            start_ms = int(existing["open_time"].max())
            logger.info("Incremental update of %s %s from %s", symbol, interval, ms_to_datetime(start_ms))
        else:
            existing = None
            start_ms = datetime_to_ms(start)
            logger.info("Full download of %s %s from %s", symbol, interval, ms_to_datetime(start_ms))

        fetched: list[list[Any]] = []
        for batch in self._source.iter_klines(symbol, interval, start_ms):
            fetched.extend(batch)
            logger.info("Fetched %d candles (up to %s)", len(fetched), ms_to_datetime(int(batch[-1][0])))

        raw = existing
        if fetched:
            new_raw = klines_to_raw_frame(fetched)
            raw = new_raw if existing is None else pd.concat([existing, new_raw], ignore_index=True)
        if raw is None:
            raw = klines_to_raw_frame([])
        raw = raw.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
        raw_path = self._storage.save_raw(raw, *key)

        as_of = self._clock()
        processed = clean_ohlcv(normalize_klines(raw), as_of)
        report = validate_ohlcv(processed, interval, as_of=as_of)

        report_path = self._storage.save_report(
            {
                "provider": self._provider,
                "symbol": symbol,
                "generated_at": as_of.isoformat(),
                **report.to_dict(),
            },
            *key,
        )

        processed_path = None
        if report.is_valid:
            processed_path = self._storage.save_processed(processed, *key)
        else:
            logger.error("Validation failed, processed dataset not updated: %s", report.errors)

        if report.gaps:
            logger.warning(
                "Dataset has %d gap(s), %d missing candle(s) in total (see %s)",
                len(report.gaps),
                report.missing_candles,
                report_path,
            )

        return CollectionResult(
            symbol=symbol,
            interval=interval,
            fetched_rows=len(fetched),
            raw_rows=len(raw),
            processed_rows=len(processed),
            report=report,
            raw_path=raw_path,
            report_path=report_path,
            processed_path=processed_path,
        )
