import pandas as pd
import pytest

from app.data.validation.market_data_validator import validate_ohlcv
from tests.factories import make_ohlcv


def test_valid_dataset_has_no_errors_or_gaps() -> None:
    report = validate_ohlcv(make_ohlcv(48), "1h")
    assert report.is_valid, report.errors
    assert report.gaps == []
    assert report.rows == 48
    assert report.first_timestamp == pd.Timestamp("2024-01-01T00:00:00Z")


def test_empty_dataset_is_invalid() -> None:
    report = validate_ohlcv(make_ohlcv(0), "1h")
    assert not report.is_valid


def test_missing_column_is_reported() -> None:
    report = validate_ohlcv(make_ohlcv(5).drop(columns="volume"), "1h")
    assert any("missing columns" in e for e in report.errors)


def test_naive_timestamps_are_rejected() -> None:
    df = make_ohlcv(5)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    report = validate_ohlcv(df, "1h")
    assert any("UTC" in e for e in report.errors)


def test_duplicates_detected() -> None:
    df = make_ohlcv(5)
    df = pd.concat([df, df.iloc[[2]]]).sort_values("timestamp").reset_index(drop=True)
    report = validate_ohlcv(df, "1h")
    assert any("duplicate" in e for e in report.errors)


def test_out_of_order_detected() -> None:
    df = make_ohlcv(5).iloc[[0, 2, 1, 3, 4]].reset_index(drop=True)
    report = validate_ohlcv(df, "1h")
    assert any("out of order" in e for e in report.errors)


def test_misaligned_timestamp_detected() -> None:
    df = make_ohlcv(5)
    df.loc[2, "timestamp"] += pd.Timedelta(minutes=30)
    report = validate_ohlcv(df, "1h")
    assert any("aligned" in e for e in report.errors)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("open", 0.0, "non-positive"),
        ("close", -5.0, "non-positive"),
        ("high", 50.0, "high below"),
        ("low", 1000.0, "low above"),
        ("volume", -1.0, "negative volume"),
        ("close", float("inf"), "infinite"),
    ],
)
def test_invalid_values_detected(column: str, value: float, message: str) -> None:
    df = make_ohlcv(5)
    df.loc[3, column] = value
    report = validate_ohlcv(df, "1h")
    assert any(message in e for e in report.errors), report.errors


def test_missing_value_detected() -> None:
    df = make_ohlcv(5)
    df.loc[1, "close"] = float("nan")
    report = validate_ohlcv(df, "1h")
    assert any("missing values in 'close'" in e for e in report.errors)


def test_gaps_are_documented_but_not_errors() -> None:
    df = make_ohlcv(10).drop(index=[4, 5, 6]).reset_index(drop=True)
    report = validate_ohlcv(df, "1h")
    assert report.is_valid
    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.missing_candles == 3
    assert gap.after == pd.Timestamp("2024-01-01T03:00:00Z")
    assert gap.before == pd.Timestamp("2024-01-01T07:00:00Z")
    assert report.to_dict()["missing_candles"] == 3


def test_unclosed_candle_detected() -> None:
    df = make_ohlcv(5)
    as_of = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=10)  # last candle still open
    report = validate_ohlcv(df, "1h", as_of=as_of.to_pydatetime())
    assert any("unclosed" in e for e in report.errors)
