"""New logic: barrier labels, metrics, training and the /barrier API."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.dependencies import get_barrier_registry
from app.api.routes.operations import OperationRequest, _command
from app.features.feature_pipeline import FeatureConfig
from app.ml.barrier.configs import BARRIER_CONFIGS, MODEL_TYPE, SIGNAL_THRESHOLD, variant_key
from app.ml.barrier.dataset import BarrierDataset, build_barrier_dataset
from app.ml.barrier.labels import barrier_outcomes
from app.ml.barrier.metrics import barrier_metrics, independent_signal_positions, pool_signals
from app.ml.barrier.training import evaluate_gate, train_barrier_model
from app.ml.models.direction_model import XGBoostDirectionModel
from app.ml.registry.model_registry import ModelRegistry
from tests.factories import make_random_ohlcv

HOUR = pd.Timedelta(hours=1)


def grid(high: list[float], low: list[float], close: float = 100.0) -> pd.DataFrame:
    n = len(high)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "high": high, "low": low, "close": [close] * n,
    })


# --- labels ----------------------------------------------------------------------------


def test_outcomes_cover_every_case() -> None:
    #       idx:  0     1      2      3     4    5    6    7      8    9    10
    high = [100, 100.5, 101.2, 100, 100, 100, 100, 101.5, 100, 100, 100]
    low = [100, 99.8, 100, 98.5, 100, 100, 100, 98.5, 100, 100, 100]
    out = barrier_outcomes(grid(high, low), threshold=0.01, max_periods=3)
    assert out["outcome"].tolist() == ["up", "up", "down", "none", "both", "both", "both", "none", "unknown", "unknown", "unknown"]
    assert out["resolve_periods"].tolist()[:8] == [2, 1, 1, 3, 3, 2, 1, 3]
    assert out["label"].tolist()[:3] == [1.0, 1.0, 0.0]
    assert out["label"].iloc[3:].isna().all()


def test_outcomes_do_not_depend_on_candles_after_resolution() -> None:
    ohlcv = make_random_ohlcv(600, seed=5)
    full = barrier_outcomes(ohlcv, 0.01, 24)
    truncated = barrier_outcomes(ohlcv.iloc[:450].reset_index(drop=True), 0.01, 24)
    known = truncated["outcome"] != "unknown"
    assert known.sum() > 300
    pd.testing.assert_frame_equal(full.iloc[:450][known.to_numpy()], truncated[known.to_numpy()])
    # the last rows of the truncated data have an unknown future
    assert (truncated["outcome"].iloc[-24:] != "unknown").sum() <= 24
    assert truncated["outcome"].iloc[-1] == "unknown"


def test_missing_candle_is_unknown_and_never_a_hit() -> None:
    high = [100, 100, 105, 100, 100]
    low = [100, 100, 95, 100, 100]
    frame = grid(high, low)
    frame.loc[2, ["high", "low", "close"]] = np.nan  # the candle with the "hit" is missing
    out = barrier_outcomes(frame, 0.01, 2)
    assert out["outcome"].tolist() == ["none", "none", "unknown", "unknown", "unknown"]


def test_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        barrier_outcomes(grid([100], [100]), 1.5, 3)
    with pytest.raises(ValueError):
        barrier_outcomes(grid([100], [100]), 0.01, 0)


# --- metrics ---------------------------------------------------------------------------


def test_independent_signals_respect_resolution_time() -> None:
    kept = independent_signal_positions(np.array([0, 1, 2, 5, 9]), np.array([3, 1, 1, 2, 1]))
    assert kept.tolist() == [0, 5, 9]


def test_barrier_metrics_split_confident_signals_by_outcome() -> None:
    outcome = np.array(["up", "up", "down", "none", "both", "down"], dtype=object)
    p = np.array([0.9, 0.8, 0.2, 0.85, 0.9, 0.6])
    durations = np.array([1, 1, 1, 3, 2, 1])
    m = barrier_metrics(outcome, p, durations, thresholds=(0.75,))
    assert m["n_resolved"] == 4 and m["resolved_share"] == pytest.approx(4 / 6)
    assert m["accuracy"] == pytest.approx(0.75)  # the 0.6 "up" was a down move
    s = m["signals"]["0.75"]
    assert s["signals"] == 5
    assert s["independent_signals"] == 3  # rows 0, 2, 4
    assert s["independent_resolved"] == 2 and s["independent_hits"] == 2
    assert s["independent_both"] == 1 and s["independent_none"] == 0
    assert s["unresolved_share"] == pytest.approx(2 / 5)


def test_pool_signals_sums_folds() -> None:
    fold = {"signals": {"0.75": {"signals": 4, "independent_signals": 3, "independent_resolved": 2, "independent_hits": 2, "independent_none": 1}}}
    pooled = pool_signals([fold, fold], 0.75)
    assert pooled["independent_resolved"] == 4 and pooled["independent_hits"] == 4
    assert pooled["independent_hit_rate"] == 1.0 and pooled["unresolved_share"] == pytest.approx(1 / 3)


def test_gate_needs_enough_confirmed_signals() -> None:
    ok = {"independent_resolved": 100, "independent_hit_rate": 0.8, "hit_rate_lower_95": 0.72}
    assert evaluate_gate(ok)["passed"]
    few = evaluate_gate({"independent_resolved": 3, "independent_hit_rate": 1.0, "hit_rate_lower_95": 0.4})
    assert not few["passed"] and "only 3" in few["reasons"][0]
    assert few["progress"]["independent_signals"] == pytest.approx(3 / 77)


# --- dataset ----------------------------------------------------------------------------


def test_dataset_keeps_unresolved_rows_without_label() -> None:
    ohlcv = make_random_ohlcv(1500, seed=2)
    dataset = build_barrier_dataset(ohlcv, "1h", BARRIER_CONFIGS["move_1pct_24h"], "candles")
    frame = dataset.frame
    assert dataset.max_periods == 24 and dataset.feature_set == "candles"
    assert len(dataset.feature_columns) > 30
    assert not (frame["outcome"] == "unknown").any()
    assert frame["label"].isna().equals(frame["outcome"].isin(["none", "both"]))
    # rows closer than 24h to the end are kept only if a barrier was already hit
    tail = frame[frame["timestamp"] > ohlcv["timestamp"].iloc[-1] - 24 * HOUR]
    assert 0 < len(tail) < 24 and tail["outcome"].isin(["up", "down", "both"]).all()
    assert frame[dataset.feature_columns].notna().all().all()
    info = dataset.describe()
    assert info["labeled_rows"] == int(frame["label"].notna().sum())
    assert pytest.approx(sum(info["outcome_shares"].values())) == 1.0


def test_dataset_rejects_unknown_feature_set() -> None:
    with pytest.raises(ValueError):
        build_barrier_dataset(make_random_ohlcv(300), "1h", BARRIER_CONFIGS["move_1pct_24h"], "sentiment")


# --- training ---------------------------------------------------------------------------


def synthetic_dataset(n: int = 5000, seed: int = 0) -> BarrierDataset:
    """Direction of the move depends on f0; one row in eight has no move at all."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p_up = 1 / (1 + np.exp(-2.5 * X[:, 0]))
    up = rng.random(n) < p_up
    outcome = np.where(up, "up", "down").astype(object)
    outcome[rng.random(n) < 0.125] = "none"
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    frame.insert(0, "timestamp", pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    frame["close"] = 100.0
    frame["outcome"] = outcome
    frame["label"] = np.where(outcome == "up", 1.0, np.where(outcome == "down", 0.0, np.nan))
    frame["resolve_periods"] = np.where(outcome == "none", 2, 1)
    config = BARRIER_CONFIGS["move_1pct_24h"]
    return BarrierDataset(frame, ["f0", "f1", "f2"], config, "candles", "1h", 2, FeatureConfig())


def test_training_reaches_target_when_the_signal_exists() -> None:
    result = train_barrier_model(synthetic_dataset(), params={"n_estimators": 150}, folds=3)
    test = result.metrics["test"]
    assert test["accuracy"] > 0.7 and test["brier_skill"] > 0.1
    wf = result.walk_forward
    assert wf["folds"] == 3 and wf["has_skill"]
    pooled = wf["pooled_signals"][f"{SIGNAL_THRESHOLD:.2f}"]
    assert pooled["independent_resolved"] >= 77
    assert 0.75 <= pooled["independent_hit_rate"] <= 0.97  # calibrated: 75%+ signals come true that often
    assert result.status == "approved"
    assert result.split_info["fit_rows_resolved"]["train"] < result.split_info["train"]["rows"]


def test_training_without_signal_stays_shadow() -> None:
    dataset = synthetic_dataset(seed=1)
    rng = np.random.default_rng(3)
    dataset.frame["f0"] = rng.normal(size=len(dataset.frame))  # break the relation
    result = train_barrier_model(dataset, params={"n_estimators": 60}, folds=3)
    assert result.status == "shadow"
    assert result.walk_forward["pooled_signals"]["0.75"]["independent_resolved"] < 77


# --- API --------------------------------------------------------------------------------


def save_barrier_model(registry: ModelRegistry, model_id: str, trained_at: str, variant: str, status: str = "shadow") -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(float)
    model = XGBoostDirectionModel({"n_estimators": 10}).fit(X, y).calibrate(X, y)
    config_key, feature_set = variant.split(":")
    config = BARRIER_CONFIGS[config_key]
    stats = {"signals": 10, "independent_signals": 4, "independent_resolved": 3, "independent_hits": 2,
             "independent_hit_rate": 2 / 3, "hit_rate_lower_95": 0.2, "independent_none": 1, "unresolved_share": 0.25}
    registry.save(model, model_id, {
        "trained_at": trained_at, "symbol": "BTCUSDT", "interval": "1h", "logic": "barrier", "horizon": config.max_horizon,
        "barrier": config.describe(), "feature_set": feature_set, "feature_set_label": feature_set, "variant": variant,
        "status": status, "signal_threshold": SIGNAL_THRESHOLD, "feature_columns": ["f0", "f1", "f2"],
        "dataset": {"rows": 300, "labeled_rows": 280, "first_timestamp": "2024-01-01T00:00:00+00:00",
                    "last_timestamp": "2024-01-13T00:00:00+00:00", "outcome_shares": {"up": 0.5, "down": 0.45, "none": 0.05, "both": 0.0}},
        "split": {"test": {"start": "2024-01-10T00:00:00+00:00", "end": "2024-01-13T00:00:00+00:00"}},
        "metrics": {"test": {"accuracy": 0.52, "majority_baseline_accuracy": 0.5, "max_confidence": 0.7, "resolved_share": 0.9,
                             "brier_skill": 0.001, "base_rate_up": 0.5, "signals": {"0.75": stats}}},
        "walk_forward": {"folds": 2, "brier_skill_by_fold": [0.01, -0.002], "brier_skill_mean": 0.004, "positive_folds": 1,
                         "has_skill": False, "accuracy_by_fold": [0.52, 0.5], "max_confidence": 0.7, "resolved_share": 0.9,
                         "by_fold": [{"fold": 1, "test_start": "a", "test_end": "b", "accuracy": 0.52, "brier_skill": 0.01,
                                      "max_confidence": 0.7, "signals": {"0.75": stats}}],
                         "pooled_signals": {"0.75": stats}, "gate": {"passed": False, "reasons": ["only 3"]}},
        "gate": {"passed": False, "reasons": ["only 3"]},
    }, model_type=MODEL_TYPE)


@pytest.fixture
def barrier_client(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("OPERATIONS_TOKEN", "secret")
    get_settings.cache_clear()
    registry = ModelRegistry(tmp_path / "barrier")
    app = create_app()
    app.dependency_overrides[get_barrier_registry] = lambda: registry
    yield TestClient(app), registry
    get_settings.cache_clear()


def test_overview_lists_configs_and_marks_latest(barrier_client) -> None:
    client, registry = barrier_client
    body = client.get("/barrier/overview").json()
    assert [c["key"] for c in body["configs"]] == list(BARRIER_CONFIGS)
    assert body["models"] == [] and body["criteria"]["gate"]["min_independent_signals"] == 77

    variant = variant_key("move_1pct_24h", "candles")
    save_barrier_model(registry, "barrier-old", "2026-01-01T00:00:00+00:00", variant)
    save_barrier_model(registry, "barrier-new", "2026-02-01T00:00:00+00:00", variant)
    save_barrier_model(registry, "barrier-other", "2026-01-15T00:00:00+00:00", variant_key("move_2pct_72h", "all_data"))
    models = client.get("/barrier/overview").json()["models"]
    assert [m["model_id"] for m in models] == ["barrier-new", "barrier-other", "barrier-old"]
    assert [m["is_latest"] for m in models] == [True, True, False]
    new = models[0]
    assert new["config_key"] == "move_1pct_24h" and new["threshold"] == 0.01 and new["feature_count"] == 3
    assert new["signals"]["independent_resolved"] == 3 and new["walk_forward"]["periods"][0]["signals"]["independent_hits"] == 2
    assert new["test"]["period_start"].startswith("2024-01-10")

    detail = client.get("/barrier/models/barrier-new").json()
    assert detail["model_type"] == MODEL_TYPE and detail["variant"] == variant
    assert client.get("/barrier/models/missing").status_code == 404


def test_delete_keeps_latest_and_requires_token(barrier_client) -> None:
    client, registry = barrier_client
    variant = variant_key("move_1pct_24h", "candles")
    save_barrier_model(registry, "barrier-old", "2026-01-01T00:00:00+00:00", variant)
    save_barrier_model(registry, "barrier-new", "2026-02-01T00:00:00+00:00", variant)

    assert client.delete("/barrier/models/barrier-old").status_code == 401
    headers = {"X-Operations-Token": "secret"}
    assert client.delete("/barrier/models/barrier-new", headers=headers).status_code == 409
    assert client.delete("/barrier/models/barrier-old", headers=headers).status_code == 200
    assert [m["model_id"] for m in client.get("/barrier/overview").json()["models"]] == ["barrier-new"]
    assert client.delete("/barrier/models/barrier-old", headers=headers).status_code == 404


def test_old_logic_registry_does_not_see_barrier_models(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    save_barrier_model(ModelRegistry(artifacts / "barrier"), "barrier-x", "2026-01-01T00:00:00+00:00", variant_key("move_1pct_24h", "candles"))
    assert ModelRegistry(artifacts).list() == []
    assert (artifacts / "barrier" / "models" / "barrier-x" / "metadata.json").exists()


def test_operations_train_barrier_command() -> None:
    request = OperationRequest(action="train_barrier", barrier_config="move_1pct_24h", barrier_feature_set="candles")
    assert _command(request) == ["scripts/train_barrier_models.py", "--config", "move_1pct_24h", "--feature-set", "candles"]
    assert _command(OperationRequest(action="train_barrier"))[2:] == ["all", "--feature-set", "all"]
    with pytest.raises(ValidationError):
        OperationRequest(action="train_barrier", barrier_config="move_9pct")
