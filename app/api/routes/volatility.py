"""Volatility ("big move") models: walk-forward quality and the current forecast."""

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends

from app.api.dependencies import _resolve, get_market_data_service, get_volatility_registry
from app.core.config import get_settings
from app.data.derivatives.store import DerivativesStore
from app.data.intervals import interval_to_timedelta
from app.features.feature_pipeline import FeatureConfig, build_features
from app.ml.evaluation.metrics import SIGNAL_THRESHOLDS
from app.ml.registry.model_registry import ModelRegistry
from app.ml.volatility.configs import FEATURE_SET_LABELS, FEATURE_SETS, MODEL_TYPE, SIGNAL_THRESHOLD, VOLATILITY_CONFIGS
from app.ml.volatility.training import GATE, SKILL_GATE
from app.services.market_data_service import MarketDataService

router = APIRouter(prefix="/volatility", tags=["volatility"])

Registry = Annotated[ModelRegistry, Depends(get_volatility_registry)]
MarketData = Annotated[MarketDataService, Depends(get_market_data_service)]


def _summary(meta: dict[str, Any]) -> dict[str, Any]:
    wf = meta.get("walk_forward") or {}
    test = (meta.get("metrics") or {}).get("test") or {}
    pooled = wf.get("pooled") or {}
    thresholds = [f"{t:.2f}" for t in SIGNAL_THRESHOLDS]
    return {
        "model_id": meta.get("model_id"),
        "trained_at": meta.get("trained_at"),
        "status": meta.get("status"),
        "config": meta.get("volatility"),
        "feature_set": meta.get("feature_set"),
        "feature_set_label": meta.get("feature_set_label"),
        "variant": meta.get("variant"),
        "dataset": meta.get("dataset"),
        "walk_forward": {
            "folds": wf.get("folds"),
            "brier_skill_by_fold": wf.get("brier_skill_by_fold"),
            "brier_skill_mean": wf.get("brier_skill_mean"),
            "baseline_brier_skill_mean": wf.get("baseline_brier_skill_mean"),
            "auc_by_fold": wf.get("auc_by_fold"),
            "positive_folds": wf.get("positive_folds"),
            "has_skill": wf.get("has_skill"),
            "beats_baseline": wf.get("beats_baseline"),
        },
        "test": {k: test.get(k) for k in ("accuracy", "majority_baseline_accuracy", "move_share", "brier_skill", "auc", "min_p_move", "max_p_move")},
        "move_signals": {t: (pooled.get("move_signals") or {}).get(t) for t in thresholds},
        "calm_signals": {t: (pooled.get("calm_signals") or {}).get(t) for t in thresholds},
        "gate": wf.get("gate"),
    }


def _latest_by_variant(registry: ModelRegistry) -> list[dict[str, Any]]:
    seen: set[str] = set()
    latest = []
    for meta in registry.list():  # newest first
        if meta.get("model_type") != MODEL_TYPE or meta.get("variant") in seen:
            continue
        seen.add(meta.get("variant"))
        latest.append(meta)
    return latest


def _forecast(registry: ModelRegistry, meta: dict[str, Any], candles: pd.DataFrame, derivatives: dict | None) -> dict[str, Any]:
    """P(big move) for the latest closed candle; unavailable if any input (e.g. derivatives) is stale."""
    config = FeatureConfig.from_dict(meta.get("feature_config") or {})
    interval = meta.get("interval", "1h")
    frame = build_features(candles, interval, config, derivatives if config.derivatives else None)
    row = frame.iloc[-1]
    columns = meta["feature_columns"]
    as_of = (row["timestamp"] + interval_to_timedelta(interval)).isoformat()
    if row[columns].isna().any():
        return {"available": False, "reason": "STALE_INPUTS", "as_of": as_of}
    model, _ = registry.load(meta["model_id"])
    p = float(model.predict_proba_up(row[columns].to_numpy(dtype="float64").reshape(1, -1))[0])
    cfg = meta.get("volatility") or {}
    return {
        "available": True,
        "as_of": as_of,
        "target_time": (row["timestamp"] + interval_to_timedelta(interval) + pd.Timedelta(cfg.get("horizon", "0h"))).isoformat(),
        "p_move": p,
        "price": float(row["close"]),
        "upper_price": float(row["close"]) * (1 + cfg.get("threshold", 0)),
        "lower_price": float(row["close"]) * (1 - cfg.get("threshold", 0)),
        "signal": "move" if p >= SIGNAL_THRESHOLD else "calm" if p <= 1 - SIGNAL_THRESHOLD else None,
    }


@router.get("/overview")
def overview(registry: Registry, market: MarketData) -> dict[str, Any]:
    models = _latest_by_variant(registry)
    forecasts: dict[str, Any] = {}
    if models:
        try:
            candles = market.get_ohlcv("BTCUSDT", "1h")
            derivatives = DerivativesStore(_resolve(get_settings().data_dir)).load_all("BTCUSDT")
            for meta in models:
                try:
                    forecasts[meta["model_id"]] = _forecast(registry, meta, candles, derivatives)
                except Exception as exc:  # one broken artifact must not hide the others
                    forecasts[meta["model_id"]] = {"available": False, "reason": type(exc).__name__}
        except Exception as exc:
            forecasts = {m["model_id"]: {"available": False, "reason": type(exc).__name__} for m in models}
    return {
        "question": "Сдвинется ли цена хотя бы на X% (в любую сторону) за N часов?",
        "signal_threshold": SIGNAL_THRESHOLD,
        "configs": [c.describe() for c in VOLATILITY_CONFIGS.values()],
        "feature_sets": [{"key": k, "label": FEATURE_SET_LABELS[k], "derivatives": list(v)} for k, v in FEATURE_SETS.items()],
        "criteria": {"gate": GATE, "skill_gate": SKILL_GATE},
        "models": [{**_summary(m), "forecast": forecasts.get(m["model_id"])} for m in models],
    }
