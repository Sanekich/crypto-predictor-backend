"""Read-only comparison data for the isolated PyTorch barrier experiment."""
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.dependencies import get_barrier_registry, get_registry
from app.ml.registry.model_registry import ModelRegistry

router = APIRouter(prefix="/pytorch", tags=["pytorch-experiment"])
Registry = Annotated[ModelRegistry, Depends(get_registry)]
BarrierRegistry = Annotated[ModelRegistry, Depends(get_barrier_registry)]


def _signals(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for key, values in (metrics.get("signals") or {}).items():
        result.append({
            "threshold": float(key),
            "signals": values.get("signals"),
            "independent_signals": values.get("independent_signals"),
            "independent_resolved": values.get("independent_resolved"),
            "independent_hits": values.get("independent_hits"),
            "independent_hit_rate": values.get("independent_hit_rate"),
            "hit_rate_lower_95": values.get("hit_rate_lower_95"),
            "independent_none": values.get("independent_none"),
        })
    return result


def _windows(meta: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    stored = meta.get("evaluation_windows") or (meta.get("walk_forward") or {}).get("by_fold") or []
    result = []
    for index, window in enumerate(stored, start=1):
        result.append({
            "fold": window.get("fold", index),
            "rows": window.get("rows"),
            "test_start": window.get("test_start"),
            "test_end": window.get("test_end"),
            "accuracy": window.get("accuracy"),
            "brier_skill": window.get("brier_skill"),
            "max_confidence": window.get("max_confidence"),
            "resolved_share": window.get("resolved_share"),
        })
    if not result:
        result.append({
            "fold": 1,
            "rows": metrics.get("n"),
            "test_start": None,
            "test_end": None,
            "accuracy": metrics.get("accuracy"),
            "brier_skill": metrics.get("brier_skill"),
            "max_confidence": metrics.get("max_confidence"),
            "resolved_share": metrics.get("resolved_share"),
        })
    return result


def _summary(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not meta:
        return None
    metrics = (meta.get("metrics") or {}).get("test") or meta.get("test") or {}
    signals = _signals(metrics)
    signal = next((item for item in signals if item["threshold"] == 0.75), {})
    windows = _windows(meta, metrics)
    skills = [item["brier_skill"] for item in windows if item.get("brier_skill") is not None]
    return {
        "model_id": meta.get("model_id"),
        "model_type": meta.get("model_type"),
        "trained_at": meta.get("trained_at"),
        "status": meta.get("status", "shadow"),
        "logic": meta.get("logic", "barrier"),
        "config_label": (meta.get("barrier") or {}).get("label", "±1% в течение 24 часов"),
        "feature_set_label": meta.get("feature_set_label", "свечи + funding + basis + open interest + DVOL"),
        "feature_count": len(meta.get("feature_columns") or []),
        "sequence_length": meta.get("sequence_length"),
        "accuracy": metrics.get("accuracy"),
        "majority_baseline_accuracy": metrics.get("majority_baseline_accuracy"),
        "brier_skill": metrics.get("brier_skill"),
        "resolved_share": metrics.get("resolved_share"),
        "max_confidence": metrics.get("max_confidence"),
        "independent_resolved": signal.get("independent_resolved"),
        "independent_hit_rate": signal.get("independent_hit_rate"),
        "hit_rate_lower_95": signal.get("hit_rate_lower_95"),
        "signals": signals,
        "windows": windows,
        "window_summary": {
            "folds": len(windows),
            "positive_folds": sum(1 for skill in skills if skill > 0),
            "mean_brier_skill": sum(skills) / len(skills) if skills else None,
        },
    }


@router.get("/overview")
def overview(registry: Registry, barrier_registry: BarrierRegistry) -> dict[str, Any]:
    pytorch = None
    for meta in registry.list():
        if meta.get("model_type") == "pytorch_gru_barrier_direction" and meta.get("symbol") == "BTCUSDT":
            pytorch = meta
            break
    xgb = None
    for meta in barrier_registry.list():
        if (
            meta.get("model_type") == "xgboost_barrier_direction"
            and meta.get("symbol") == "BTCUSDT"
            and meta.get("variant") == "move_1pct_24h:all_data"
        ):
            xgb = meta
            break
    return {
        "target": "Какой барьер будет достигнут первым: +1% или -1% в течение 24 часов",
        "feature_set": "свечи + funding + basis + open interest + DVOL",
        "pytorch": _summary(pytorch),
        "xgboost": _summary(xgb),
    }
