import numpy as np
import pandas as pd
import pytest

from app.data.processing.resampling import reindex_to_grid
from app.features.feature_pipeline import build_dataset, build_features, feature_columns
from app.features.targets import horizon_to_periods
from tests.factories import make_random_ohlcv

N = 600


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return make_random_ohlcv(N, seed=42)


def test_no_lookahead_features_are_identical_on_truncated_history(ohlcv) -> None:
    """Features at time t must not change when future candles are removed."""
    full = build_features(ohlcv, "1h")
    columns = feature_columns(full)
    for cut in (200, 350, 599):
        truncated = build_features(ohlcv.iloc[:cut], "1h")
        pd.testing.assert_series_equal(
            truncated[columns].iloc[cut - 1],
            full[columns].iloc[cut - 1],
            check_names=False,
        )


def test_no_lookahead_by_perturbing_future(ohlcv) -> None:
    """Changing future prices must not affect past features."""
    base = build_features(ohlcv, "1h")
    changed = ohlcv.copy()
    changed.loc[400:, ["open", "high", "low", "close"]] *= 3
    changed.loc[400:, "volume"] *= 10
    perturbed = build_features(changed, "1h")
    columns = feature_columns(base)
    pd.testing.assert_frame_equal(base[columns].iloc[:400], perturbed[columns].iloc[:400])


def test_target_is_future_return(ohlcv) -> None:
    dataset = build_dataset(ohlcv, "1h", "24h")
    frame = dataset.frame.set_index("timestamp")
    close = ohlcv.set_index("timestamp")["close"]
    for ts in frame.index[[0, 50, -1]]:
        expected = close[ts + pd.Timedelta(hours=24)] / close[ts] - 1
        assert frame.loc[ts, dataset.target_return_column] == pytest.approx(expected)
        assert frame.loc[ts, dataset.target_direction_column] == float(expected > 0)


def test_target_never_used_as_feature(ohlcv) -> None:
    dataset = build_dataset(ohlcv, "1h", "24h")
    assert not any(col.startswith("target_") for col in dataset.feature_columns)
    assert "close" not in dataset.feature_columns


def test_dataset_has_no_nan_or_inf(ohlcv) -> None:
    dataset = build_dataset(ohlcv, "1h", "24h")
    values = dataset.frame[dataset.feature_columns + [dataset.target_return_column]].to_numpy()
    assert np.isfinite(values).all()
    assert len(dataset.frame) > 300


def test_last_horizon_rows_are_excluded(ohlcv) -> None:
    dataset = build_dataset(ohlcv, "1h", "24h")
    assert dataset.frame["timestamp"].max() == ohlcv["timestamp"].iloc[-1] - pd.Timedelta(hours=24)


def test_feature_order_is_deterministic(ohlcv) -> None:
    a = build_dataset(ohlcv, "1h", "24h").feature_columns
    b = build_dataset(ohlcv.iloc[:500], "1h", "24h").feature_columns
    assert a == b
    assert len(a) == len(set(a))


def test_features_are_scale_free(ohlcv) -> None:
    """Multiplying all prices by a constant must not change features."""
    scaled = ohlcv.copy()
    scaled[["open", "high", "low", "close"]] *= 1000
    a = build_features(ohlcv, "1h")
    b = build_features(scaled, "1h")
    columns = feature_columns(a)
    pd.testing.assert_frame_equal(a[columns], b[columns], rtol=1e-9)


def test_gap_does_not_create_fake_returns(ohlcv) -> None:
    with_gap = ohlcv.drop(index=range(300, 305)).reset_index(drop=True)
    features = build_features(with_gap, "1h")

    assert len(features) == N  # missing candles restored as empty rows
    row_after_gap = features.index[features["timestamp"] == ohlcv["timestamp"].iloc[305]][0]
    assert np.isnan(features.loc[row_after_gap, "log_return_1"])
    assert features.loc[300:304, "close"].isna().all()

    dataset = build_dataset(with_gap, "1h", "24h")
    gap_start, gap_end = ohlcv["timestamp"].iloc[300], ohlcv["timestamp"].iloc[304]
    in_gap = dataset.frame["timestamp"].between(gap_start, gap_end)
    assert not in_gap.any()
    # Targets pointing into the gap are unknown and dropped too
    targets_into_gap = dataset.frame["timestamp"].between(
        gap_start - pd.Timedelta(hours=24), gap_end - pd.Timedelta(hours=24)
    )
    assert not targets_into_gap.any()


def test_reindex_to_grid_fills_missing_rows(ohlcv) -> None:
    sparse = ohlcv.drop(index=[5, 6]).reset_index(drop=True)
    grid = reindex_to_grid(sparse, "1h")
    assert len(grid) == len(ohlcv)
    assert grid["timestamp"].equals(ohlcv["timestamp"])
    assert grid.loc[[5, 6], "close"].isna().all()


@pytest.mark.parametrize(
    ("horizon", "interval", "expected"),
    [("24h", "1h", 24), ("1d", "4h", 6), ("4h", "1h", 4), ("1w", "1d", 7), ("30m", "15m", 2)],
)
def test_horizon_to_periods(horizon, interval, expected) -> None:
    assert horizon_to_periods(horizon, interval) == expected


@pytest.mark.parametrize(("horizon", "interval"), [("90m", "1h"), ("24", "1h"), ("0h", "1h"), ("1h", "4h")])
def test_invalid_horizon(horizon, interval) -> None:
    with pytest.raises(ValueError):
        horizon_to_periods(horizon, interval)
