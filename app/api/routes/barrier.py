"""New-logic (barrier) models: progress toward the 75% target, for the monitoring page.

These models live in artifacts/barrier and are not used for live forecasts yet;
the endpoints only read their saved evaluation (test period + walk-forward).
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.dependencies import get_barrier_registry
from app.api.routes.operations import _check_token
from app.core.exceptions import NotFoundError
from app.ml.barrier.configs import BARRIER_CONFIGS, FEATURE_SET_LABELS, FEATURE_SETS, MODEL_TYPE, SIGNAL_THRESHOLD
from app.ml.barrier.training import GATE, SKILL_GATE
from app.ml.evaluation.metrics import SIGNAL_THRESHOLDS
from app.ml.registry.model_registry import ModelNotFoundError, ModelRegistry

router = APIRouter(prefix="/barrier", tags=["barrier"])

Registry = Annotated[ModelRegistry, Depends(get_barrier_registry)]

_SIGNAL_KEYS = ("signals", "independent_signals", "independent_resolved", "independent_hits", "independent_hit_rate",
                "hit_rate_lower_95", "independent_none", "unresolved_share")


def _signal_summary(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if not stats else {k: stats.get(k) for k in _SIGNAL_KEYS}


def _summary(meta: dict[str, Any]) -> dict[str, Any]:
    key = f"{SIGNAL_THRESHOLD:.2f}"
    wf = meta.get("walk_forward") or {}
    pooled = wf.get("pooled_signals") or {}
    test = (meta.get("metrics") or {}).get("test") or {}
    dataset = meta.get("dataset") or {}
    barrier = meta.get("barrier") or {}
    test_period = (meta.get("split") or {}).get("test") or {}
    return {
        "model_id": meta.get("model_id"),
        "trained_at": meta.get("trained_at"),
        "status": meta.get("status"),
        "symbol": meta.get("symbol"),
        "interval": meta.get("interval"),
        "config_key": barrier.get("key"),
        "config_label": barrier.get("label"),
        "threshold": barrier.get("threshold"),
        "max_horizon": barrier.get("max_horizon"),
        "feature_set": meta.get("feature_set"),
        "feature_set_label": meta.get("feature_set_label"),
        "variant": meta.get("variant"),
        "feature_count": len(meta.get("feature_columns") or []),
        "rows": dataset.get("rows"),
        "labeled_rows": dataset.get("labeled_rows"),
        "first_timestamp": dataset.get("first_timestamp"),
        "last_timestamp": dataset.get("last_timestamp"),
        "outcome_shares": dataset.get("outcome_shares"),
        "test": {
            "period_start": test_period.get("start"),
            "period_end": test_period.get("end"),
            "accuracy": test.get("accuracy"),
            "majority_baseline_accuracy": test.get("majority_baseline_accuracy"),
            "base_rate_up": test.get("base_rate_up"),
            "max_confidence": test.get("max_confidence"),
            "resolved_share": test.get("resolved_share"),
            "brier_skill": test.get("brier_skill"),
            "signals": _signal_summary((test.get("signals") or {}).get(key)),
        },
        "walk_forward": {
            "folds": wf.get("folds"),
            "brier_skill_by_fold": wf.get("brier_skill_by_fold"),
            "brier_skill_mean": wf.get("brier_skill_mean"),
            "positive_folds": wf.get("positive_folds"),
            "has_skill": wf.get("has_skill"),
            "accuracy_by_fold": wf.get("accuracy_by_fold"),
            "max_confidence": wf.get("max_confidence"),
            "resolved_share": wf.get("resolved_share"),
            "periods": [
                {
                    "fold": m.get("fold"),
                    "test_start": m.get("test_start"),
                    "test_end": m.get("test_end"),
                    "accuracy": m.get("accuracy"),
                    "brier_skill": m.get("brier_skill"),
                    "max_confidence": m.get("max_confidence"),
                    "signals": _signal_summary((m.get("signals") or {}).get(key)),
                }
                for m in wf.get("by_fold") or []
            ],
        },
        "signals": _signal_summary(pooled.get(key)),
        "signals_by_threshold": {t: _signal_summary(pooled.get(t)) for t in (f"{x:.2f}" for x in SIGNAL_THRESHOLDS) if t in pooled},
        "gate": meta.get("gate"),
    }


def _models(registry: ModelRegistry) -> list[dict[str, Any]]:
    """All barrier models, newest first; the newest of each variant is marked latest."""
    items = [m for m in registry.list() if m.get("model_type") == MODEL_TYPE]
    latest: dict[str, str] = {}
    for meta in items:
        latest.setdefault(str(meta.get("variant")), meta["model_id"])
    return [{**_summary(m), "is_latest": latest.get(str(m.get("variant"))) == m["model_id"]} for m in items]


@router.get("/overview")
def overview(registry: Registry) -> dict[str, Any]:
    return {
        "logic": {
            "name": "barrier",
            "question": "Что наступит раньше: рост цены на X% или падение на X%?",
            "signal_threshold": SIGNAL_THRESHOLD,
            "artifacts_dir": str(registry.models_dir.parent.name),
        },
        "configs": [c.describe() for c in BARRIER_CONFIGS.values()],
        "feature_sets": [{"key": k, "label": FEATURE_SET_LABELS[k], "derivatives": list(v)} for k, v in FEATURE_SETS.items()],
        "criteria": {"gate": GATE, "skill_gate": SKILL_GATE},
        "models": _models(registry),
    }


@router.get("/models/{model_id}")
def get_model(model_id: str, registry: Registry) -> dict[str, Any]:
    """Full saved metadata of a barrier model (dataset, split, every fold and threshold)."""
    try:
        meta = registry.metadata(model_id)
    except (ModelNotFoundError, ValueError):
        raise NotFoundError(f"Model {model_id} not found") from None
    if meta.get("model_type") != MODEL_TYPE:
        raise NotFoundError(f"Model {model_id} not found")
    return meta


@router.delete("/models/{model_id}")
def delete_model(
    model_id: str,
    registry: Registry,
    x_operations_token: str | None = Header(default=None),
    x_operations_token_b64: str | None = Header(default=None),
) -> dict[str, str]:
    """Delete an old version of a barrier experiment; the newest version of each variant is kept."""
    _check_token(x_operations_token, x_operations_token_b64)
    models = {m["model_id"]: m for m in _models(registry)}
    meta = models.get(model_id)
    if meta is None:
        raise NotFoundError(f"Model {model_id} not found")
    if meta["status"] == "approved":
        raise HTTPException(status_code=409, detail="Approved models cannot be deleted")
    if meta["is_latest"]:
        raise HTTPException(status_code=409, detail="The latest version of this experiment cannot be deleted")
    registry.delete(model_id)
    return {"model_id": model_id, "status": "deleted"}
