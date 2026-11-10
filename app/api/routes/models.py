from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.dependencies import get_registry
from app.core.exceptions import NotFoundError
from app.ml.registry.model_registry import ModelNotFoundError, ModelRegistry
from app.schemas.prediction import ModelSummary

router = APIRouter(prefix="/models", tags=["models"])

Registry = Annotated[ModelRegistry, Depends(get_registry)]


@router.get("", response_model=list[ModelSummary])
def list_models(registry: Registry) -> list[dict[str, Any]]:
    return registry.list()


@router.get("/{model_id}")
def get_model(model_id: str, registry: Registry) -> dict[str, Any]:
    """Full metadata of a model version: features, periods, metrics, baselines."""
    try:
        return registry.metadata(model_id)
    except (ModelNotFoundError, ValueError):
        raise NotFoundError(f"Model {model_id} not found") from None
