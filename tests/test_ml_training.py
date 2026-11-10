import json

import numpy as np
import pandas as pd
import pytest

from app.ml.evaluation.metrics import regression_metrics
from app.ml.evaluation.splits import chronological_split
from app.ml.models.baselines import HistoricalMeanBaseline, RidgeBaseline, ZeroReturnBaseline
from app.ml.models.xgboost_model import XGBoostReturnModel
from app.ml.registry.model_registry import ModelNotFoundError, ModelRegistry, make_model_id
from app.ml.training.train import train_and_evaluate

HOUR = pd.Timedelta(hours=1)


def hourly(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))


def synthetic_frame(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Target depends non-linearly on two features plus noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 0.01 * X[:, 0] + 0.01 * np.sign(X[:, 1]) * X[:, 1] ** 2 + rng.normal(0, 0.005, n)
    columns = [f"f{i}" for i in range(5)]
    frame = pd.DataFrame(X, columns=columns)
    frame.insert(0, "timestamp", hourly(n))
    frame["target"] = y
    return frame, columns


# --- split -------------------------------------------------------------------


def test_split_is_chronological_with_gap() -> None:
    ts = hourly(1000)
    gap = 24 * HOUR
    split = chronological_split(ts, gap)

    train_ts, val_ts, test_ts = ts[split.train], ts[split.validation], ts[split.test]
    assert train_ts.max() < val_ts.min() and val_ts.max() < test_ts.min()
    # Purging: the training target window (t, t + gap] ends before validation starts
    assert train_ts.max() + gap < val_ts.min()
    assert val_ts.max() + gap < test_ts.min()
    assert set(split.train).isdisjoint(split.validation) and set(split.validation).isdisjoint(split.test)
    assert len(split.test) == 150


def test_split_rejects_unsorted() -> None:
    with pytest.raises(ValueError):
        chronological_split(hourly(100).iloc[::-1], HOUR)


def test_split_rejects_too_small_data() -> None:
    with pytest.raises(ValueError):
        chronological_split(hourly(20), 24 * HOUR)


# --- metrics -----------------------------------------------------------------


def test_regression_metrics_known_values() -> None:
    m = regression_metrics(np.array([0.01, -0.02, 0.03]), np.array([0.02, -0.01, -0.01]))
    assert m["mae"] == pytest.approx(0.02)
    assert m["mae_bps"] == pytest.approx(200)
    assert m["rmse"] == pytest.approx(np.sqrt((0.01**2 + 0.01**2 + 0.04**2) / 3))
    assert m["direction_accuracy"] == pytest.approx(2 / 3)


def test_constant_forecast_has_no_correlation() -> None:
    m = regression_metrics(np.array([0.01, -0.01, 0.02]), np.zeros(3))
    assert m["correlation"] is None
    assert m["direction_accuracy"] == pytest.approx(1 / 3)  # predicts "not up"


# --- models ------------------------------------------------------------------


def test_baselines() -> None:
    X = np.zeros((4, 2))
    y = np.array([0.01, 0.03, -0.01, 0.01])
    assert ZeroReturnBaseline().fit(X, y).predict(X).tolist() == [0, 0, 0, 0]
    assert HistoricalMeanBaseline().fit(X, y).predict(X[:2]) == pytest.approx([0.01, 0.01])


def test_xgboost_learns_signal_and_beats_baselines() -> None:
    frame, columns = synthetic_frame()
    result = train_and_evaluate(frame, columns, "target", gap=24 * HOUR)

    rmse = {e.name: e.metrics["test"]["rmse"] for e in result.evaluations}
    assert rmse["xgboost"] < rmse["ridge"] < rmse["zero_return"]
    assert result.best_by_validation.name == "xgboost"
    assert result.best_baseline().name == "ridge"
    assert result.get("xgboost").model.training_info()["best_iteration"] is not None


def test_models_never_see_validation_or_test_targets() -> None:
    """Corrupting validation/test targets must not change the fitted baseline."""
    frame, columns = synthetic_frame(1000)
    clean = train_and_evaluate(frame, columns, "target", 24 * HOUR, models=[HistoricalMeanBaseline(), RidgeBaseline()])

    corrupted = frame.copy()
    corrupted.loc[clean.split.validation[0] :, "target"] = 1.0
    dirty = train_and_evaluate(corrupted, columns, "target", 24 * HOUR, models=[HistoricalMeanBaseline(), RidgeBaseline()])

    assert clean.get("historical_mean").model.mean_ == dirty.get("historical_mean").model.mean_
    X = frame[columns].to_numpy()
    np.testing.assert_allclose(clean.get("ridge").model.predict(X), dirty.get("ridge").model.predict(X))


def test_feature_importance_finds_informative_features() -> None:
    frame, columns = synthetic_frame()
    result = train_and_evaluate(frame, columns, "target", 24 * HOUR, models=[XGBoostReturnModel()])
    importance = result.get("xgboost").model.feature_importance(columns)
    assert set(list(importance)[:2]) == {"f0", "f1"}
    assert sum(importance.values()) == pytest.approx(1.0)


# --- range models ------------------------------------------------------------


def heteroscedastic_frame(n: int = 4000, seed: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """Return volatility depends on feature f0: a good range model must widen with it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    sigma = 0.005 + 0.01 * np.abs(X[:, 0])
    y = rng.normal(0, sigma)
    columns = ["f0", "f1", "f2"]
    frame = pd.DataFrame(X, columns=columns)
    frame.insert(0, "timestamp", hourly(n))
    frame["target"] = y
    return frame, columns


def test_interval_metrics_known_values() -> None:
    from app.ml.evaluation.metrics import interval_metrics, pinball_loss

    y = np.array([0.0, 0.05, -0.05, 0.01])
    preds = np.tile([-0.02, 0.0, 0.02], (4, 1))
    m = interval_metrics(y, preds, (0.1, 0.5, 0.9))
    assert m["coverage"] == pytest.approx(0.5)
    assert m["mean_width"] == pytest.approx(0.04)
    assert m["nominal_coverage"] == pytest.approx(0.8)
    assert m["width_correlation"] is None
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.9) == pytest.approx(0.1)


def test_quantile_model_adapts_range_and_beats_baseline() -> None:
    from app.ml.training.train_range import train_range_models

    frame, columns = heteroscedastic_frame()
    result = train_range_models(frame, columns, "target", 24 * HOUR)
    model, baseline = result.get("xgboost_quantile"), result.get("historical_quantiles")

    test = model.metrics["test"]
    assert test["pinball_mean"] < baseline.metrics["test"]["pinball_mean"]
    assert 0.7 < test["coverage"] < 0.9
    assert test["width_correlation"] > 0.3

    quantiles = model.model.predict_quantiles(frame[columns].to_numpy()[:100])
    assert quantiles.shape == (100, 3)
    assert (np.diff(quantiles, axis=1) >= 0).all()  # no crossing


def test_quantile_model_roundtrip(tmp_path) -> None:
    frame, columns = heteroscedastic_frame(600)
    X, y = frame[columns].to_numpy(), frame["target"].to_numpy()
    from app.ml.models.quantile_model import XGBoostQuantileModel

    model = XGBoostQuantileModel(params={"n_estimators": 20}).fit(X, y)
    registry = ModelRegistry(tmp_path)
    registry.save(model, "xgboost-quantile-test", {})
    loaded, meta = registry.load("xgboost-quantile-test")
    assert meta["model_type"] == "xgboost_quantile"
    assert loaded.quantiles == (0.1, 0.5, 0.9)
    np.testing.assert_allclose(loaded.predict_quantiles(X), model.predict_quantiles(X), rtol=1e-6)


# --- registry ----------------------------------------------------------------


def test_registry_roundtrip(tmp_path) -> None:
    frame, columns = synthetic_frame(800)
    X, y = frame[columns].to_numpy(), frame["target"].to_numpy()
    model = XGBoostReturnModel({"n_estimators": 50}).fit(X[:600], y[:600], X[600:], y[600:])

    registry = ModelRegistry(tmp_path)
    model_id = make_model_id("xgboost", "BTCUSDT", "1h", "24h")
    registry.save(model, model_id, {"feature_columns": columns, "trained_at": "2026-01-01", "score": float("nan")})

    loaded, meta = registry.load(model_id)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X), rtol=1e-6)
    assert meta["model_type"] == "xgboost"
    assert meta["feature_columns"] == columns
    assert meta["score"] is None  # NaN stored as valid JSON null
    json.loads((tmp_path / "models" / model_id / "metadata.json").read_text(encoding="utf-8"))
    assert [m["model_id"] for m in registry.list()] == [model_id]

    with pytest.raises(FileExistsError):
        registry.save(model, model_id, {})


def test_registry_baseline_roundtrip(tmp_path) -> None:
    X, y = np.ones((5, 2)), np.arange(5.0)
    registry = ModelRegistry(tmp_path)
    registry.save(HistoricalMeanBaseline().fit(X, y), "historical_mean-test", {})
    loaded, _ = registry.load("historical_mean-test")
    assert loaded.predict(X[:1]) == pytest.approx([2.0])


def test_registry_rejects_bad_ids(tmp_path) -> None:
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.metadata("../secret")
    with pytest.raises(ModelNotFoundError):
        registry.metadata("missing-model")


def test_model_id_format() -> None:
    model_id = make_model_id("xgboost", "BTCUSDT", "1h", "24h", pd.Timestamp("2026-11-10 12:30:00", tz="UTC"))
    assert model_id == "xgboost-btcusdt-1h-h24h-20261110-123000"
