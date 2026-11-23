"""Volatility ("big move") models: labels, metrics, training and the /volatility API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_market_data_service, get_volatility_registry
from app.api.routes.operations import OperationRequest, _command
from app.features.feature_pipeline import FeatureConfig
from app.ml.registry.model_registry import ModelRegistry
from app.ml.volatility.configs import MODEL_TYPE, VOLATILITY_CONFIGS, variant_key
from app.ml.volatility.dataset import BASELINE_FEATURE, VolatilityDataset, build_volatility_dataset
from app.ml.volatility.metrics import pool, volatility_metrics
from app.ml.volatility.training import train_volatility_model
from tests.factories import make_random_ohlcv

HOUR = pd.Timedelta(hours=1)


# --- dataset -----------------------------------------------------------------------------


def test_label_is_any_barrier_touch_within_horizon() -> None:
    ohlcv = make_random_ohlcv(800, seed=4)
    config = VOLATILITY_CONFIGS["move_1pct_4h"]
    dataset = build_volatility_dataset(ohlcv, "1h", config, "candles")
    frame = dataset.frame.set_index("timestamp")
    source = ohlcv.set_index("timestamp")
    for ts in frame.index[100:160]:
        close = source.loc[ts, "close"]
        future = source.loc[ts + HOUR : ts + 4 * HOUR]
        moved = (future["high"] >= close * 1.01).any() or (future["low"] <= close * 0.99).any()
        assert frame.loc[ts, "label"] == float(moved)
    assert dataset.periods == 4
    assert frame[dataset.feature_columns].notna().all().all()
    assert BASELINE_FEATURE in dataset.feature_columns
    assert 0 < dataset.describe()["move_share"] < 1


def test_rows_without_known_future_are_dropped() -> None:
    ohlcv = make_random_ohlcv(600, seed=1)
    dataset = build_volatility_dataset(ohlcv, "1h", VOLATILITY_CONFIGS["move_3pct_24h"], "candles")
    tail = dataset.frame[dataset.frame["timestamp"] > ohlcv["timestamp"].iloc[-1] - 24 * HOUR]
    assert (tail["label"] == 1).all()  # only rows whose move already happened are known early


# --- metrics -----------------------------------------------------------------------------


def test_metrics_count_move_and_calm_signals() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.6, 0.3, 0.1, 0.85])
    m = volatility_metrics(y, p, spacing=1, thresholds=(0.75,))
    move, calm = m["move_signals"]["0.75"], m["calm_signals"]["0.75"]
    assert move["independent_signals"] == 3 and move["independent_hits"] == 2  # 0.9, 0.8 right; 0.85 wrong
    assert calm["independent_signals"] == 3 and calm["independent_hits"] == 2  # 0.1, 0.2 right; 0.1 wrong
    assert 0 < m["auc"] < 1 and m["move_share"] == 0.5


def test_perfect_ranking_has_auc_one_and_spacing_is_respected() -> None:
    y = np.array([0, 0, 1, 1])
    m = volatility_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), spacing=2, thresholds=(0.75,))
    assert m["auc"] == 1.0
    assert m["move_signals"]["0.75"]["independent_signals"] == 1  # rows 2 and 3 are one move
    pooled = pool([m, m], "move_signals", 0.75)
    assert pooled["independent_signals"] == 2 and pooled["independent_hit_rate"] == 1.0


# --- training ----------------------------------------------------------------------------


def synthetic(n: int = 5000, seed: int = 0, informative: bool = True) -> VolatilityDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    vol = np.exp(rng.normal(-4, 0.3, n))
    logit = 2.5 * X[:, 0] if informative else np.zeros(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    frame[BASELINE_FEATURE] = vol
    frame.insert(0, "timestamp", pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    frame["close"] = 100.0
    frame["label"] = y
    frame["resolve_periods"] = 1
    columns = ["f0", "f1", "f2", BASELINE_FEATURE]
    return VolatilityDataset(frame, columns, VOLATILITY_CONFIGS["move_1pct_4h"], "candles", "1h", 2, FeatureConfig())


def test_training_finds_skill_and_beats_naive_baseline() -> None:
    result = train_volatility_model(synthetic(), params={"n_estimators": 150}, folds=3)
    wf = result.walk_forward
    assert wf["folds"] == 3 and wf["has_skill"] and wf["beats_baseline"]
    assert wf["brier_skill_mean"] > 0.1 > abs(wf["baseline_brier_skill_mean"])
    move = wf["pooled"]["move_signals"]["0.75"]
    assert move["independent_signals"] > 77 and 0.75 <= move["independent_hit_rate"] <= 0.97
    assert result.status == "approved"


def test_training_without_signal_stays_shadow() -> None:
    result = train_volatility_model(synthetic(seed=2, informative=False), params={"n_estimators": 60}, folds=3)
    assert result.status == "shadow" and not result.walk_forward["gate"]["passed"]


# --- API ---------------------------------------------------------------------------------


class StubMarket:
    def __init__(self, ohlcv: pd.DataFrame) -> None:
        self.ohlcv = ohlcv

    def get_ohlcv(self, symbol: str, interval: str, refresh: bool = True) -> pd.DataFrame:
        return self.ohlcv


def test_overview_returns_quality_and_current_forecast(tmp_path) -> None:
    from app.main import create_app

    ohlcv = make_random_ohlcv(1200, seed=9)
    for column in ("open", "high", "low", "close"):  # calmer market: both classes occur
        ohlcv[column] = 100 * (ohlcv[column] / 100) ** 0.3
    config = VOLATILITY_CONFIGS["move_1pct_4h"]
    dataset = build_volatility_dataset(ohlcv, "1h", config, "candles")
    result = train_volatility_model(dataset, params={"n_estimators": 30}, folds=2)
    registry = ModelRegistry(tmp_path / "volatility")
    registry.save(result.model, "volatility-test", {
        "trained_at": "2026-01-01T00:00:00+00:00", "symbol": "BTCUSDT", "interval": "1h", "horizon": "4h",
        "volatility": config.describe(), "feature_set": "candles", "feature_set_label": "candles",
        "variant": variant_key(config.key, "candles"), "status": result.status, "walk_forward": result.walk_forward,
        "feature_columns": dataset.feature_columns, "feature_config": dataset.feature_config.to_dict(),
        "dataset": dataset.describe(), "metrics": result.metrics,
    }, model_type=MODEL_TYPE)

    app = create_app()
    app.dependency_overrides[get_volatility_registry] = lambda: registry
    app.dependency_overrides[get_market_data_service] = lambda: StubMarket(ohlcv)
    body = TestClient(app).get("/volatility/overview").json()
    assert [c["key"] for c in body["configs"]] == list(VOLATILITY_CONFIGS)
    model = body["models"][0]
    assert model["model_id"] == "volatility-test" and model["walk_forward"]["folds"] == 2
    forecast = model["forecast"]
    assert forecast["available"] and 0 < forecast["p_move"] < 1
    assert forecast["upper_price"] == pytest.approx(forecast["price"] * 1.01)
    assert forecast["as_of"] == (ohlcv["timestamp"].iloc[-1] + HOUR).isoformat()


def test_operations_train_volatility_command() -> None:
    assert _command(OperationRequest(action="train_volatility", barrier_feature_set="candles")) == [
        "scripts/train_volatility_models.py", "--feature-set", "candles"]
