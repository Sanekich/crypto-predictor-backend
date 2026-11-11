import json
from contextlib import contextmanager
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_prediction_service, get_registry
from app.core.exceptions import DataStaleError, ModelIncompatibleError, ModelNotAvailableError
from app.data.processing.cleaning import OHLCV_COLUMNS
from app.data.storage import MarketDataStorage
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION, FeatureConfig, build_dataset, build_features
from app.main import app
from app.ml.models.quantile_model import XGBoostQuantileModel
from app.ml.registry.model_registry import ModelRegistry
from app.services.market_data_service import MarketDataService
from app.services.prediction_service import FEATURE_HISTORY_ROWS, PredictionService
from tests.factories import START_MS, make_klines, make_random_ohlcv

N = 700
LAST_OPEN = pd.Timestamp(START_MS, unit="ms", tz="UTC") + pd.Timedelta(hours=N - 1)
FRESH_NOW = (LAST_OPEN + pd.Timedelta(minutes=70)).to_pydatetime()  # last candle closed 10 min ago


class NoNetworkSource:
    calls = 0

    def iter_klines(self, *args, **kwargs):
        raise AssertionError("network must not be used")


@contextmanager
def no_network():
    yield NoNetworkSource()


def save_model(registry: ModelRegistry, ohlcv: pd.DataFrame, horizon: str = "24h", **overrides) -> str:
    dataset = build_dataset(ohlcv, "1h", horizon)
    X = dataset.frame[dataset.feature_columns].to_numpy()
    y = dataset.frame[dataset.target_return_column].to_numpy()
    model = XGBoostQuantileModel(params={"n_estimators": 30}).fit(X, y)
    model_id = f"xgboost_quantile-test-{horizon}"
    metadata = {
        "trained_at": "2026-01-01T00:00:00+00:00",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "horizon": horizon,
        "horizon_periods": dataset.horizon_periods,
        "feature_columns": dataset.feature_columns,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "feature_config": json.loads(json.dumps(FeatureConfig().to_dict())),
        "split": {"test": {"start": "2024-01-20T00:00:00+00:00", "end": "2024-01-29T00:00:00+00:00"}},
        "metrics": {"test": {"nominal_coverage": 0.8, "coverage": 0.81}},
        "improvement_vs_baseline": {"pinball": {"test": 0.05}},
        **overrides,
    }
    registry.save(model, model_id, metadata)
    return model_id


@pytest.fixture
def env(tmp_path):
    ohlcv = make_random_ohlcv(N, seed=3)
    storage = MarketDataStorage(tmp_path / "data")
    storage.save_processed(ohlcv, "binance", "BTCUSDT", "1h")
    registry = ModelRegistry(tmp_path / "artifacts")
    save_model(registry, ohlcv)
    market = MarketDataService(storage, no_network, clock=lambda: FRESH_NOW)
    service = PredictionService(registry, market, clock=lambda: FRESH_NOW)
    return {"ohlcv": ohlcv, "storage": storage, "registry": registry, "service": service}


# --- prediction service -------------------------------------------------------


def test_prediction_is_consistent(env) -> None:
    result = env["service"].predict("btcusdt", "24h")
    current = env["ohlcv"]["close"].iloc[-1]

    assert result.symbol == "BTCUSDT"
    assert result.current_price == pytest.approx(current)
    assert result.range.lower_return <= result.median_return <= result.range.upper_return
    assert result.range.lower_price == pytest.approx(current * (1 + result.range.lower_return))
    assert result.range.probability == pytest.approx(0.8)
    assert result.data_as_of == LAST_OPEN + pd.Timedelta(hours=1)
    assert result.target_time == LAST_OPEN + pd.Timedelta(hours=25)
    assert result.direction.available is False and result.direction.reason == "NO_MODEL"
    assert result.quality.test_coverage == pytest.approx(0.81)


def test_prediction_is_cached_per_candle(env) -> None:
    service = env["service"]
    assert service.predict("BTCUSDT", "24h") is service.predict("BTCUSDT", "24h")


def test_direction_reports_no_model_without_horizon_direction_model(env) -> None:
    from app.ml.models.baselines import ZeroReturnBaseline

    env["registry"].save(
        ZeroReturnBaseline(),
        "xgboost-test",
        {
            "trained_at": "2026-01-01",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "horizon": "24h",
            "improvement_vs_best_baseline": {"validation_rmse": -0.001},
        },
    )
    # registry.latest matches on model_type; patch the stored type to "xgboost"
    meta_path = env["registry"].models_dir / "xgboost-test" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["model_type"] = "xgboost"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert env["service"].predict("BTCUSDT", "24h").direction.reason == "NO_MODEL"


def test_unknown_horizon_raises(env) -> None:
    with pytest.raises(ModelNotAvailableError):
        env["service"].predict("BTCUSDT", "4h")


def test_stale_data_raises(env) -> None:
    late = FRESH_NOW + timedelta(hours=5)
    market = MarketDataService(env["storage"], no_network, clock=lambda: late, min_refresh_interval=timedelta(days=1))
    market._last_attempt[("BTCUSDT", "1h")] = late  # refresh throttled -> stays stale
    service = PredictionService(env["registry"], market, clock=lambda: late)
    with pytest.raises(DataStaleError):
        service.predict("BTCUSDT", "24h")


def test_incompatible_pipeline_version_raises(tmp_path, env) -> None:
    registry = ModelRegistry(tmp_path / "other")
    save_model(registry, env["ohlcv"], feature_pipeline_version="0.1")
    market = MarketDataService(env["storage"], no_network, clock=lambda: FRESH_NOW)
    with pytest.raises(ModelIncompatibleError):
        PredictionService(registry, market, clock=lambda: FRESH_NOW).predict("BTCUSDT", "24h")


def test_list_assets(env) -> None:
    save_model(env["registry"], env["ohlcv"], horizon="7d")
    assets = env["service"].list_assets()
    assert len(assets) == 1
    assert assets[0].name == "Bitcoin"
    assert assets[0].horizons == ["24h", "7d"]


def test_features_from_recent_history_match_full_history() -> None:
    """Inference uses only the last rows; the result must equal full-history features."""
    ohlcv = make_random_ohlcv(FEATURE_HISTORY_ROWS + 1500, seed=5)
    full = build_features(ohlcv, "1h").iloc[-1]
    tail = build_features(ohlcv.iloc[-FEATURE_HISTORY_ROWS:], "1h").iloc[-1]
    columns = [c for c in full.index if c != "timestamp"]
    np.testing.assert_allclose(tail[columns].to_numpy(float), full[columns].to_numpy(float), rtol=1e-9, atol=1e-12)


# --- market data service -------------------------------------------------------


def test_market_data_refreshes_when_new_candle_closed(tmp_path) -> None:
    storage = MarketDataStorage(tmp_path)
    klines = make_klines(50)
    from app.data.collection.collector import MarketDataCollector

    first_now = pd.Timestamp(START_MS, unit="ms", tz="UTC") + pd.Timedelta(hours=40, minutes=5)

    class Source:
        def iter_klines(self, symbol, interval, start_ms, end_ms=None):
            yield [k for k in klines if k[0] >= start_ms]

    @contextmanager
    def factory():
        yield Source()

    MarketDataCollector(Source(), storage, clock=lambda: first_now.to_pydatetime()).collect(
        "BTCUSDT", "1h", pd.Timestamp(START_MS, unit="ms", tz="UTC").to_pydatetime()
    )
    assert len(storage.load_processed("binance", "BTCUSDT", "1h")) == 40

    later = (first_now + pd.Timedelta(hours=5)).to_pydatetime()
    service = MarketDataService(storage, factory, clock=lambda: later)
    df = service.get_ohlcv("BTCUSDT", "1h")
    assert len(df) == 45
    assert list(df.columns) == OHLCV_COLUMNS


def test_market_data_refresh_failure_keeps_existing_data(env) -> None:
    @contextmanager
    def failing():
        raise OSError("network down")
        yield  # pragma: no cover

    late = FRESH_NOW + timedelta(hours=3)
    service = MarketDataService(env["storage"], failing, clock=lambda: late)
    assert len(service.get_ohlcv("BTCUSDT", "1h")) == N


# --- API -------------------------------------------------------------------------


@pytest.fixture
def client(env):
    app.dependency_overrides[get_prediction_service] = lambda: env["service"]
    app.dependency_overrides[get_registry] = lambda: env["registry"]
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_prediction(client) -> None:
    response = client.get("/predictions/BTCUSDT", params={"horizon": "24h"})
    assert response.status_code == 200
    body = response.json()
    assert body["range"]["lower_price"] < body["range"]["upper_price"]
    assert body["direction"]["available"] is False
    assert "Not financial advice" in body["disclaimer"]


def test_api_structured_error(client) -> None:
    response = client.get("/predictions/BTCUSDT", params={"horizon": "4h"})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "MODEL_NOT_AVAILABLE"


def test_api_validates_input(client) -> None:
    assert client.get("/predictions/BTCUSDT", params={"horizon": "tomorrow"}).status_code == 422
    assert client.get("/predictions/BTC-USDT").status_code == 422


def test_api_assets_and_models(client) -> None:
    assert client.get("/assets").json()[0]["symbol"] == "BTCUSDT"
    models = client.get("/models").json()
    assert models[0]["model_type"] == "xgboost_quantile"
    assert client.get(f"/models/{models[0]['model_id']}").json()["horizon"] == "24h"
    assert client.get("/models/nope").json()["detail"]["code"] == "NOT_FOUND"


def test_api_cors_allows_frontend(client) -> None:
    response = client.get("/assets", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/assets", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers
