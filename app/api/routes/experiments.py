"""Read-only overview of isolated experimental models."""
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.dependencies import get_experiment_registry
from app.ml.registry.model_registry import ModelRegistry

router = APIRouter(prefix="/experiments", tags=["experiments"])
ExperimentRegistry = Annotated[ModelRegistry, Depends(get_experiment_registry)]


def _summary(meta: dict[str, Any]) -> dict[str, Any]:
    metrics = (meta.get("metrics") or {}).get("test") or {}
    walk_forward = meta.get("walk_forward") or {}
    signal = (walk_forward.get("pooled_signals") or {}).get("0.75", {})
    folds = walk_forward.get("by_fold") or []
    skills = [row.get("brier_skill") for row in folds if row.get("brier_skill") is not None]
    return {
        "model_id": meta.get("model_id"),
        "model_type": meta.get("model_type"),
        "trained_at": meta.get("trained_at"),
        "status": meta.get("status", "shadow"),
        "logic": meta.get("logic", "experimental"),
        "barrier_label": (meta.get("barrier") or {}).get("label", "+/-1% within 24h"),
        "feature_set": meta.get("feature_set", "candles"),
        "feature_set_label": meta.get("feature_set_label", "candles + state/shock"),
        "feature_count": len(meta.get("feature_columns") or []),
        "rows": (meta.get("dataset") or {}).get("rows"),
        "first_timestamp": (meta.get("dataset") or {}).get("first_timestamp"),
        "last_timestamp": (meta.get("dataset") or {}).get("last_timestamp"),
        "accuracy": metrics.get("accuracy"),
        "baseline_accuracy": metrics.get("majority_baseline_accuracy"),
        "brier_skill": metrics.get("brier_skill"),
        "max_confidence": metrics.get("max_confidence"),
        "independent_resolved": signal.get("independent_resolved"),
        "independent_hit_rate": signal.get("independent_hit_rate"),
        "hit_rate_lower_95": signal.get("hit_rate_lower_95"),
        "folds": len(folds),
        "positive_folds": sum(1 for value in skills if value > 0),
        "mean_brier_skill": sum(skills) / len(skills) if skills else walk_forward.get("brier_skill_mean"),
    }


@router.get("/overview")
def overview(registry: ExperimentRegistry) -> dict[str, Any]:
    models = [
        _summary(meta)
        for meta in registry.list()
        if meta.get("model_type") in {"xgboost_state_aware_barrier", "pytorch_gru_barrier_direction"}
        and meta.get("symbol", "BTCUSDT") == "BTCUSDT"
    ]
    return {"models": models[:20], "latest": models[0] if models else None}
