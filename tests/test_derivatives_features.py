import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import ModelIncompatibleError
from app.features.derivatives import _hourly_metrics, asof_values, build_derivative_features
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION, FeatureConfig, build_features
from app.ml.evaluation.walk_forward import expanding_folds
from app.services.features_service import check_compatibility
from tests.factories import make_random_ohlcv

N = 900
HOUR = pd.Timedelta(hours=1)
ALL = ("funding", "basis", "open_interest", "dvol")


def synthetic_derivatives(ohlcv: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start, end = ohlcv["timestamp"].iloc[0], ohlcv["timestamp"].iloc[-1] + HOUR
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    funding_times = pd.date_range(start, end, freq="8h", tz="UTC")
    five_min = pd.date_range(start, end, freq="5min", tz="UTC")
    return {
        "funding": pd.DataFrame({"available_at": funding_times, "funding_rate": rng.normal(1e-4, 5e-5, len(funding_times))}),
        "premium": pd.DataFrame({
            "timestamp": hours, "available_at": hours + HOUR,
            "premium_close": rng.normal(0, 5e-4, len(hours)), "premium_high": 0.0, "premium_low": 0.0,
        }),
        "metrics": pd.DataFrame({
            "available_at": five_min,
            "open_interest": 1.0,
            "open_interest_value": 1e9 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(five_min)))),
            "top_trader_account_ratio": 1.0,
            "top_trader_position_ratio": rng.uniform(0.8, 1.5, len(five_min)),
            "global_account_ratio": rng.uniform(0.8, 2.0, len(five_min)),
            "taker_buy_sell_ratio": rng.uniform(0.5, 1.5, len(five_min)),
        }),
        "dvol": pd.DataFrame({"timestamp": hours, "available_at": hours + HOUR, "dvol": rng.uniform(35, 80, len(hours))}),
    }


@pytest.fixture(scope="module")
def data():
    ohlcv = make_random_ohlcv(N, seed=7)
    return ohlcv, synthetic_derivatives(ohlcv)


def deriv_columns(frame: pd.DataFrame) -> list[str]:
    base = set(build_features(make_random_ohlcv(400), "1h").columns)
    return [c for c in frame.columns if c not in base]


def test_derivative_features_have_no_lookahead(data) -> None:
    """Changing everything published after time T must not change features of candles closing <= T."""
    ohlcv, deriv = data
    config = FeatureConfig(derivatives=ALL)
    base = build_features(ohlcv, "1h", config, deriv)

    cut = ohlcv["timestamp"].iloc[600] + HOUR  # close of candle 600
    changed = {}
    for name, frame in deriv.items():
        frame = frame.copy()
        later = frame["available_at"] > cut
        numeric = [c for c in frame.columns if c not in ("timestamp", "available_at")]
        frame.loc[later, numeric] = frame.loc[later, numeric] * 3 + 1
        changed[name] = frame
    perturbed_ohlcv = ohlcv.copy()
    perturbed_ohlcv.loc[601:, ["open", "high", "low", "close"]] *= 2

    after = build_features(perturbed_ohlcv, "1h", config, changed)
    columns = deriv_columns(base)
    assert len(columns) == 16
    pd.testing.assert_frame_equal(base[columns].iloc[:601], after[columns].iloc[:601])
    assert not base[columns].iloc[700:].equals(after[columns].iloc[700:])


def test_value_published_after_close_is_not_used() -> None:
    close_times = pd.Series(pd.to_datetime(["2024-01-01 01:00", "2024-01-01 02:00"], utc=True))
    frame = pd.DataFrame({
        "available_at": pd.to_datetime(["2024-01-01 00:30:00", "2024-01-01 01:00:01"], utc=True),
        "x": [1.0, 2.0],
    })
    values = asof_values(close_times, frame, ["x"], pd.Timedelta(hours=2))["x"].tolist()
    assert values == [1.0, 2.0]  # 01:00:01 is too late for the 01:00 close


def test_outage_produces_nan_not_stale_value() -> None:
    close_times = pd.Series(pd.to_datetime(["2024-01-01 01:00", "2024-01-01 05:00"], utc=True))
    frame = pd.DataFrame({"available_at": pd.to_datetime(["2024-01-01 00:59"], utc=True), "x": [1.0]})
    values = asof_values(close_times, frame, ["x"], pd.Timedelta(minutes=30))["x"]
    assert values.iloc[0] == 1.0 and np.isnan(values.iloc[1])


def test_hourly_metrics_need_enough_snapshots() -> None:
    times = list(pd.date_range("2024-01-01 00:05", periods=12, freq="5min", tz="UTC"))  # full hour 00:00-01:00
    times += [pd.Timestamp("2024-01-01 01:05", tz="UTC")]  # only one snapshot in the next hour
    m = pd.DataFrame({
        "available_at": times, "open_interest_value": range(13), "top_trader_position_ratio": 1.0,
        "global_account_ratio": 1.0, "taker_buy_sell_ratio": 1.0,
    })
    hourly = _hourly_metrics(m)
    assert hourly["available_at"].tolist() == [pd.Timestamp("2024-01-01 01:00", tz="UTC")]
    assert hourly["open_interest_value"].iloc[0] == 11  # last snapshot of the hour


def test_missing_source_is_reported(data) -> None:
    ohlcv, deriv = data
    with pytest.raises(ValueError, match="dvol"):
        build_derivative_features(ohlcv[["timestamp", "close"]], {k: v for k, v in deriv.items() if k != "dvol"}, ("dvol",), HOUR)


def test_unknown_group_rejected() -> None:
    with pytest.raises(ValueError):
        FeatureConfig(derivatives=("sentiment",))


# --- compatibility -------------------------------------------------------------------


def meta_with(config: dict) -> dict:
    return {"model_id": "m", "feature_pipeline_version": FEATURE_PIPELINE_VERSION, "feature_config": config}


def test_old_model_metadata_without_derivatives_key_is_compatible() -> None:
    old = json.loads(json.dumps(FeatureConfig().to_dict()))
    old.pop("derivatives")
    check_compatibility(meta_with(old), FeatureConfig())  # must not raise


def test_derivatives_model_rejected_at_inference() -> None:
    stored = json.loads(json.dumps(FeatureConfig(derivatives=("funding",)).to_dict()))
    with pytest.raises(ModelIncompatibleError, match="derivatives"):
        check_compatibility(meta_with(stored), FeatureConfig())


# --- walk-forward gaps ---------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 24, 168])
def test_walk_forward_folds_are_purged(horizon) -> None:
    ts = pd.Series(pd.date_range("2020-01-01", periods=20_000, freq="h", tz="UTC"))
    folds = expanding_folds(ts, horizon, folds=4)
    assert folds
    for f in folds:
        assert f.train[-1] + horizon < f.early_stopping[0]
        assert f.early_stopping[-1] + horizon < f.calibration[0]
        # calibration targets (t, t + horizon] must end before the test period starts
        assert f.calibration[-1] + horizon <= f.test[0]


# --- operations API --------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from app.core.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("OPERATIONS_TOKEN", "secret")
    get_settings.cache_clear()
    yield TestClient(create_app())
    get_settings.cache_clear()


def test_operations_log_requires_token(client) -> None:
    assert client.get("/operations/status").status_code == 401
    assert client.get("/operations/status", headers={"X-Operations-Token": "wrong"}).status_code == 401
    assert client.get("/operations/status", headers={"X-Operations-Token": "secret"}).status_code == 200


def test_cors_allows_delete_from_frontend(client) -> None:
    response = client.options(
        "/models/some-model",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "DELETE"},
    )
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]


def test_training_stage_requires_two_months_of_live_data() -> None:
    from app.api.routes.training_stage import LIVE_REQUIRED_HOURS, _model_stage

    assert LIVE_REQUIRED_HOURS >= 24 * 60
    assert _model_stage({}) == 1
    assert _model_stage({"derivatives": ["funding", "basis"]}) == 2
    assert _model_stage({"derivatives": ["funding", "basis", "dvol"]}) == 3
