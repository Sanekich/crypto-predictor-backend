import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.data.collection.collector import MarketDataCollector
from app.data.storage import MarketDataStorage
from tests.factories import HOUR_MS, START_MS, make_klines

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


class FakeSource:
    """Serves a fixed list of klines in batches, like the Binance client."""

    def __init__(self, klines: list[list]) -> None:
        self.klines = klines
        self.requested_from: list[int] = []

    def iter_klines(self, symbol, interval, start_ms, end_ms=None):
        self.requested_from.append(start_ms)
        rows = [k for k in self.klines if k[0] >= start_ms]
        for i in range(0, len(rows), 100):
            yield rows[i : i + 100]


def clock_at(hours_after_start: float):
    return lambda: START + timedelta(hours=hours_after_start)


def test_full_download_saves_raw_processed_and_report(tmp_path) -> None:
    storage = MarketDataStorage(tmp_path)
    source = FakeSource(make_klines(250))
    # 249.5h after start: the last candle (open at 249h) is not closed yet
    collector = MarketDataCollector(source, storage, clock=clock_at(249.5))

    result = collector.collect("btcusdt", "1h", START)

    assert result.symbol == "BTCUSDT"
    assert result.fetched_rows == 250
    assert result.raw_rows == 250
    assert result.processed_rows == 249
    assert result.report.is_valid

    processed = storage.load_processed("binance", "BTCUSDT", "1h")
    assert len(processed) == 249
    assert str(processed["timestamp"].dt.tz) == "UTC"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["symbol"] == "BTCUSDT"
    assert report["is_valid"] is True
    assert report["rows"] == 249


def test_incremental_update_fetches_only_new_candles(tmp_path) -> None:
    storage = MarketDataStorage(tmp_path)
    klines = make_klines(300)

    MarketDataCollector(FakeSource(klines[:200]), storage, clock=clock_at(199.5)).collect("BTCUSDT", "1h", START)

    source = FakeSource(klines)
    result = MarketDataCollector(source, storage, clock=clock_at(300)).collect("BTCUSDT", "1h", START)

    # Resumes from the last stored candle (it was unclosed during the first run)
    assert source.requested_from == [START_MS + 199 * HOUR_MS]
    assert result.fetched_rows == 101
    assert result.raw_rows == 300
    assert result.processed_rows == 300
    assert result.report.is_valid


def test_full_refresh_ignores_existing_data(tmp_path) -> None:
    storage = MarketDataStorage(tmp_path)
    klines = make_klines(50)
    MarketDataCollector(FakeSource(klines), storage, clock=clock_at(100)).collect("BTCUSDT", "1h", START)

    source = FakeSource(klines)
    MarketDataCollector(source, storage, clock=clock_at(100)).collect("BTCUSDT", "1h", START, full_refresh=True)

    assert source.requested_from == [START_MS]


def test_invalid_data_does_not_overwrite_processed(tmp_path) -> None:
    storage = MarketDataStorage(tmp_path)
    good = make_klines(20)
    MarketDataCollector(FakeSource(good), storage, clock=clock_at(100)).collect("BTCUSDT", "1h", START)

    bad = make_klines(30)
    bad[25][2] = "1.00000000"  # high below open/close
    result = MarketDataCollector(FakeSource(bad), storage, clock=clock_at(100)).collect("BTCUSDT", "1h", START)

    assert not result.report.is_valid
    assert result.processed_path is None
    assert len(storage.load_processed("binance", "BTCUSDT", "1h")) == 20
    assert json.loads(result.report_path.read_text(encoding="utf-8"))["is_valid"] is False


def test_gaps_are_reported(tmp_path) -> None:
    storage = MarketDataStorage(tmp_path)
    klines = make_klines(30)
    del klines[10:13]

    result = MarketDataCollector(FakeSource(klines), storage, clock=clock_at(100)).collect("BTCUSDT", "1h", START)

    assert result.report.is_valid
    assert result.report.missing_candles == 3
    processed = storage.load_processed("binance", "BTCUSDT", "1h")
    assert pd.Timestamp("2024-01-01T10:00:00Z") not in set(processed["timestamp"])
