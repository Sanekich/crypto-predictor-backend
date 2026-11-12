from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.dependencies import get_registry
from app.core.exceptions import NotFoundError
from app.ml.registry.model_registry import ModelNotFoundError, ModelRegistry
from app.api.routes.operations import _check_token
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


@router.delete("/{model_id}")
def delete_old_model(
    model_id: str,
    registry: Registry,
    x_operations_token: str | None = Header(default=None),
    x_operations_token_b64: str | None = Header(default=None),
) -> dict[str, str]:
    """Delete only a historical shadow model; protected by OPERATIONS_TOKEN."""
    _check_token(x_operations_token, x_operations_token_b64)
    try:
        registry.delete_old(model_id)
    except ModelNotFoundError:
        raise NotFoundError(f"Model {model_id} not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"model_id": model_id, "status": "deleted"}
