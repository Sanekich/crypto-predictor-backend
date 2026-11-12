import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_signal_service
from app.features.feature_pipeline import FEATURE_PIPELINE_VERSION, FeatureConfig, build_dataset
from app.main import app
from app.ml.evaluation.metrics import direction_metrics, independent_positions, wilson_lower_bound
from app.ml.models.direction_model import XGBoostDirectionModel
from app.ml.registry.model_registry import ModelRegistry
from app.ml.training.train_direction import evaluate_gate, train_direction_model
from app.services.signal_journal import SignalJournal
from app.services.signal_service import SignalService
from tests.factories import make_random_ohlcv
from tests.test_prediction_service import FRESH_NOW, LAST_OPEN, N, no_network
from app.data.storage import MarketDataStorage
from app.services.market_data_service import MarketDataService

HOUR = pd.Timedelta(hours=1)


# --- metrics ------------------------------------------------------------------------


def test_wilson_lower_bound_known_values() -> None:
    assert wilson_lower_bound(8, 10) == pytest.approx(0.490, abs=0.001)
    assert wilson_lower_bound(80, 100) == pytest.approx(0.711, abs=0.001)
    assert wilson_lower_bound(0, 0) is None


def test_independent_positions_skip_overlapping_signals() -> None:
    assert independent_positions(np.array([0, 1, 2, 4, 5, 9]), 4).tolist() == [0, 4, 9]
    assert independent_positions(np.array([3, 4, 5]), 1).tolist() == [3, 4, 5]


def test_direction_metrics_counts_signals() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.55, 0.2, 0.5])
    m = direction_metrics(y, p, horizon_periods=1, thresholds=(0.75,))
    s = m["signals"]["0.75"]
    # confident rows: 0.9 (up, right), 0.8 (up, right), 0.2 (down, wrong)
    assert s["signals"] == 3
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert m["base_rate_up"] == pytest.approx(0.5)
    assert m["max_confidence"] == pytest.approx(0.9)


def test_gate() -> None:
    def metrics(n, hit, low):
        return {"signals": {"0.75": {"independent_signals": n, "independent_hit_rate": hit, "hit_rate_lower_95": low}}}

    assert evaluate_gate(metrics(100, 0.8, 0.72))["passed"]
    failed = evaluate_gate(metrics(0, None, None))
    assert not failed["passed"] and "only 0" in failed["reasons"][0]
    assert not evaluate_gate(metrics(300, 0.7, 0.65))["passed"]


# --- model ------------------------------------------------------------------------


def signal_frame(n: int = 6000, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Direction depends on f0: P(up) = sigmoid(2 * f0)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-2 * X[:, 0]))
    y = (rng.random(n) < p).astype(float)
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    frame.insert(0, "timestamp", pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    frame["up"] = y
    return frame, ["f0", "f1", "f2"]


def test_calibrated_probabilities_match_frequencies() -> None:
    frame, columns = signal_frame()
    result = train_direction_model(frame, columns, "up", horizon_periods=1, gap=HOUR)
    test = result.metrics["test"]

    assert test["accuracy"] > 0.65
    assert test["brier"] < test["brier_base_rate"]
    s = test["signals"]["0.75"]
    assert s["signals"] > 50
    # Calibrated: signals with >=75% confidence come true roughly that often
    assert 0.75 <= s["hit_rate"] <= 0.95
    assert test["max_confidence"] < 0.999


def test_direction_model_roundtrip(tmp_path) -> None:
    frame, columns = signal_frame(1500)
    X, y = frame[columns].to_numpy(), frame["up"].to_numpy()
    model = XGBoostDirectionModel({"n_estimators": 30}).fit(X[:1000], y[:1000]).calibrate(X[1000:], y[1000:])
    registry = ModelRegistry(tmp_path)
    registry.save(model, "xgboost-direction-test", {})
    loaded, meta = registry.load("xgboost-direction-test")
    assert meta["model_type"] == "xgboost_direction"
    np.testing.assert_allclose(loaded.predict_proba_up(X), model.predict_proba_up(X), rtol=1e-6)


# --- service ----------------------------------------------------------------------


def save_direction_model(
    registry: ModelRegistry, ohlcv: pd.DataFrame, horizon: str, threshold: float, status: str, has_skill: bool = True
) -> None:
    dataset = build_dataset(ohlcv, "1h", horizon)
    X = dataset.frame[dataset.feature_columns].to_numpy()
    y = dataset.frame[dataset.target_direction_column].to_numpy()
    model = XGBoostDirectionModel({"n_estimators": 20}).fit(X, y).calibrate(X, y)
    registry.save(
        model,
        f"xgboost-direction-test-{horizon}",
        {
            "trained_at": "2026-01-01T00:00:00+00:00",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "horizon": horizon,
            "horizon_periods": dataset.horizon_periods,
            "status": status,
            "signal_threshold": threshold,
            "gate": {"reasons": [] if status == "approved" else ["only 0 independent signals"]},
            "walk_forward": {"has_skill": has_skill, "folds": 5, "positive_folds": 5 if has_skill else 1, "brier_skill_mean": 0.01 if has_skill else -0.02},
            "feature_columns": dataset.feature_columns,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "feature_config": json.loads(json.dumps(FeatureConfig().to_dict())),
            "split": {"test": {"start": "2024-01-20T00:00:00+00:00", "end": "2024-01-29T00:00:00+00:00"}},
            "metrics": {
                "test": {
                    "accuracy": 0.52,
                    "always_up_accuracy": 0.5,
                    "max_confidence": 0.6,
                    "signals": {f"{threshold:.2f}": {"independent_signals": 0, "independent_hit_rate": None, "hit_rate_lower_95": None}},
                }
            },
        },
    )


@pytest.fixture
def signal_env(tmp_path):
    ohlcv = make_random_ohlcv(N, seed=3)
    storage = MarketDataStorage(tmp_path / "data")
    storage.save_processed(ohlcv, "binance", "BTCUSDT", "1h")
    registry = ModelRegistry(tmp_path / "artifacts")
    # threshold 0.5 -> every prediction is a "signal", so the journal gets entries
    save_direction_model(registry, ohlcv, "4h", threshold=0.5, status="shadow")
    save_direction_model(registry, ohlcv, "1h", threshold=0.5, status="approved")
    market = MarketDataService(storage, no_network, clock=lambda: FRESH_NOW)
    journal = SignalJournal(tmp_path / "signals")
    service = SignalService(registry, market, journal, clock=lambda: FRESH_NOW)
    return {"service": service, "journal": journal, "ohlcv": ohlcv}


def test_signals_report_all_horizons_sorted(signal_env) -> None:
    result = signal_env["service"].get_signals("btcusdt", threshold=0.75)
    assert [h.horizon for h in result.horizons] == ["1h", "4h"]
    one_hour, four_hours = result.horizons
    assert one_hour.status == "approved" and four_hours.status == "shadow"
    assert four_hours.status_reasons == ["only 0 independent signals"]
    for h in result.horizons:
        assert 0 <= h.p_up <= 1 and h.confidence == pytest.approx(max(h.p_up, 1 - h.p_up))
        assert h.direction == ("up" if h.p_up >= 0.5 else "down")
    assert four_hours.target_time == LAST_OPEN + 5 * HOUR
    assert result.data_as_of == LAST_OPEN + HOUR


def test_signals_are_journaled_once_per_candle(signal_env) -> None:
    service = signal_env["service"]
    service.get_signals("BTCUSDT")
    service.get_signals("BTCUSDT")
    journal = signal_env["journal"].load("BTCUSDT")
    assert len(journal) == 2  # one per horizon, not duplicated by the second request
    assert set(journal["model_status"]) == {"approved", "shadow"}


def test_journal_outcomes_are_resolved_from_market_data(tmp_path) -> None:
    ohlcv = make_random_ohlcv(100, seed=1)
    journal = SignalJournal(tmp_path)
    t0 = ohlcv["timestamp"].iloc[10] + HOUR  # signal at the close of candle 10
    price = float(ohlcv["close"].iloc[10])
    later = float(ohlcv["close"].iloc[14])  # close at t0 + 4h
    journal.record("BTCUSDT", {"horizon": "4h", "signal_time": t0, "target_time": t0 + 4 * HOUR, "direction": "up",
                               "p_up": 0.8, "confidence": 0.8, "price_at_signal": price, "model_version": "m", "model_status": "shadow"})
    journal.record("BTCUSDT", {"horizon": "4h", "signal_time": ohlcv["timestamp"].iloc[-1] + HOUR,
                               "target_time": ohlcv["timestamp"].iloc[-1] + 5 * HOUR, "direction": "down",
                               "p_up": 0.2, "confidence": 0.8, "price_at_signal": 1.0, "model_version": "m", "model_status": "shadow"})

    result = journal.with_outcomes("BTCUSDT", ohlcv, "1h").sort_values("signal_time")
    resolved, pending = result.iloc[0], result.iloc[1]
    assert resolved["outcome_price"] == pytest.approx(later)
    assert bool(resolved["correct"]) == (later > price)
    assert pd.isna(pending["outcome_price"]) and pd.isna(pending["correct"])


def test_signals_api(signal_env) -> None:
    app.dependency_overrides[get_signal_service] = lambda: signal_env["service"]
    try:
        client = TestClient(app)
        body = client.get("/signals/BTCUSDT", params={"min_confidence": 0.75}).json()
        assert body["threshold"] == 0.75
        assert len(body["horizons"]) == 2
        assert body["live"]["signals"] == 2
        assert client.get("/signals/BTCUSDT", params={"min_confidence": 0.3}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_no_signal_without_walk_forward_skill(tmp_path) -> None:
    ohlcv = make_random_ohlcv(N, seed=3)
    storage = MarketDataStorage(tmp_path / "data")
    storage.save_processed(ohlcv, "binance", "BTCUSDT", "1h")
    registry = ModelRegistry(tmp_path / "artifacts")
    save_direction_model(registry, ohlcv, "4h", threshold=0.5, status="shadow", has_skill=False)
    market = MarketDataService(storage, no_network, clock=lambda: FRESH_NOW)
    service = SignalService(registry, market, SignalJournal(tmp_path / "signals"), clock=lambda: FRESH_NOW)

    h = service.get_signals("BTCUSDT", threshold=0.5).horizons[0]
    assert h.has_skill is False and h.walk_forward_positive_folds == 1
    assert h.is_signal is False  # even at threshold 0.5: no skill -> never a signal


def test_prediction_direction_hidden_without_skill(tmp_path) -> None:
    from app.services.prediction_service import PredictionService
    from tests.test_prediction_service import save_model

    ohlcv = make_random_ohlcv(N, seed=3)
    storage = MarketDataStorage(tmp_path / "data")
    storage.save_processed(ohlcv, "binance", "BTCUSDT", "1h")
    registry = ModelRegistry(tmp_path / "artifacts")
    save_model(registry, ohlcv, horizon="24h")
    market = MarketDataService(storage, no_network, clock=lambda: FRESH_NOW)

    save_direction_model(registry, ohlcv, "24h", threshold=0.75, status="shadow", has_skill=False)
    direction = PredictionService(registry, market, clock=lambda: FRESH_NOW).predict("BTCUSDT", "24h").direction
    assert direction.available is False and direction.reason == "NO_SIGNAL"
