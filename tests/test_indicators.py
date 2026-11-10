import numpy as np
import pandas as pd
import pytest

from app.features import technical_indicators as ti
from app.features.returns import log_return, simple_return
from app.features.volume import taker_buy_ratio, volume_ratio


def test_returns() -> None:
    close = pd.Series([100.0, 110.0, 99.0])
    assert simple_return(close).tolist()[1:] == pytest.approx([0.1, -0.1])
    assert log_return(close, 2).iloc[2] == pytest.approx(np.log(0.99))
    assert np.isnan(simple_return(close).iloc[0])


def test_sma_requires_full_window() -> None:
    result = ti.sma(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
    assert result.isna().tolist() == [True, True, False, False]
    assert result.iloc[3] == pytest.approx(3.0)


def test_ema_matches_recursive_definition() -> None:
    values = pd.Series([10.0, 11.0, 12.0, 13.0, 12.0])
    span = 3
    alpha = 2 / (span + 1)
    expected = [values.iloc[0]]
    for v in values.iloc[1:]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])
    result = ti.ema(values, span)
    assert result.iloc[2:].tolist() == pytest.approx(expected[2:])
    assert result.iloc[:2].isna().all()


def test_rsi_extremes_and_range() -> None:
    rising = pd.Series(np.arange(1.0, 40.0))
    assert ti.rsi(rising, 14).dropna().eq(100.0).all()

    falling = pd.Series(np.arange(40.0, 1.0, -1))
    assert ti.rsi(falling, 14).dropna().eq(0.0).all()

    flat = pd.Series([5.0] * 30)
    assert ti.rsi(flat, 14).dropna().eq(50.0).all()

    noisy = pd.Series(100 + np.random.default_rng(1).normal(0, 1, 300).cumsum())
    values = ti.rsi(noisy, 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_rsi_known_value() -> None:
    # Gains of 1 and losses of 1 alternating -> average gain == average loss -> RSI 50
    close = pd.Series([10.0, 11.0] * 20)
    assert ti.rsi(close, 14).iloc[-1] == pytest.approx(50.0, abs=5)


def test_macd_histogram_is_line_minus_signal() -> None:
    close = pd.Series(100 + np.random.default_rng(2).normal(0, 1, 200).cumsum())
    line, signal, hist = ti.macd(close)
    pd.testing.assert_series_equal(hist, line - signal)
    assert line.iloc[:25].isna().all()


def test_true_range_uses_previous_close() -> None:
    high = pd.Series([10.0, 12.0])
    low = pd.Series([9.0, 11.0])
    close = pd.Series([9.5, 11.5])
    tr = ti.true_range(high, low, close)
    assert tr.tolist() == pytest.approx([1.0, 2.5])  # gap up: 12 - 9.5


def test_volume_features_handle_zero_volume() -> None:
    vol = pd.Series([0.0, 0.0, 0.0, 10.0])
    assert np.isnan(volume_ratio(vol, 3).iloc[2])  # mean 0 -> NaN, not inf
    assert np.isnan(taker_buy_ratio(pd.Series([0.0]), pd.Series([0.0])).iloc[0])
