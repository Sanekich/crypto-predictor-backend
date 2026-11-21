from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_barrier_registry, get_experiment_registry, get_registry
from app.api.routes.operations import _check_token
from app.ml.registry.model_registry import ModelNotFoundError, ModelRegistry

router = APIRouter(prefix="/model-history", tags=["model-history"])
Registry = Annotated[ModelRegistry, Depends(get_registry)]
BarrierRegistry = Annotated[ModelRegistry, Depends(get_barrier_registry)]
ExperimentRegistry = Annotated[ModelRegistry, Depends(get_experiment_registry)]


class ArchiveRequest(BaseModel):
    model_id: str
    source: Literal["base", "barrier", "experiment"]
    reason: str = Field(min_length=3, max_length=300)


def _registries(registry: ModelRegistry, barrier: ModelRegistry, experiments: ModelRegistry) -> dict[str, ModelRegistry]:
    return {"base": registry, "barrier": barrier, "experiment": experiments}


def _entry(meta: dict[str, Any], source: str, archived: bool) -> dict[str, Any]:
    test = meta.get("metrics", {}).get("test", {})
    walk = meta.get("walk_forward", {})
    return {
        "model_id": meta.get("model_id"),
        "source": source,
        "archived": archived,
        "model_type": meta.get("model_type"),
        "symbol": meta.get("symbol"),
        "interval": meta.get("interval"),
        "horizon": meta.get("horizon"),
        "feature_set": meta.get("feature_set"),
        "status": meta.get("status"),
        "trained_at": meta.get("trained_at"),
        "archived_at": meta.get("archived_at"),
        "archive_reason": meta.get("archive_reason"),
        "accuracy": test.get("accuracy"),
        "baseline_accuracy": test.get("majority_baseline_accuracy", test.get("always_up_accuracy")),
        "brier": test.get("brier"),
        "brier_skill": walk.get("brier_skill_mean"),
        "positive_folds": walk.get("positive_folds"),
        "folds": walk.get("folds"),
        "independent_signals": test.get("signals", {}).get("0.75", {}).get("independent_signals"),
        "metadata": meta,
    }


@router.get("")
def history(registry: Registry, barrier: BarrierRegistry, experiments: ExperimentRegistry) -> dict[str, Any]:
    sources = _registries(registry, barrier, experiments)
    active = [_entry(meta, source, False) for source, store in sources.items() for meta in store.list()]
    archived = [_entry(meta, source, True) for source, store in sources.items() for meta in store.list_archived()]
    active.sort(key=lambda item: item.get("trained_at") or "", reverse=True)
    archived.sort(key=lambda item: item.get("archived_at") or "", reverse=True)
    return {"active": active, "archived": archived}


@router.post("/archive")
def archive(
    request: ArchiveRequest,
    registry: Registry,
    barrier: BarrierRegistry,
    experiments: ExperimentRegistry,
    x_operations_token: str | None = Header(default=None),
    x_operations_token_b64: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_token(x_operations_token, x_operations_token_b64)
    store = _registries(registry, barrier, experiments)[request.source]
    try:
        meta = store.archive_old(request.model_id, request.reason)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found") from None
    except (ValueError, FileExistsError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _entry(meta, request.source, True)
