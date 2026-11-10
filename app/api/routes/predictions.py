from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import get_prediction_service
from app.schemas.prediction import AssetInfo, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter(tags=["predictions"])

Service = Annotated[PredictionService, Depends(get_prediction_service)]


@router.get("/assets", response_model=list[AssetInfo])
def list_assets(service: Service) -> list[AssetInfo]:
    """Assets and horizons for which a forecast model is available."""
    return service.list_assets()


@router.get("/predictions/{symbol}", response_model=PredictionResponse)
def get_prediction(
    service: Service,
    symbol: Annotated[str, Path(pattern=r"^[A-Za-z0-9]{2,20}$", examples=["BTCUSDT"])],
    horizon: Annotated[str, Query(pattern=r"^\d+[mhdw]$", examples=["24h", "7d"])] = "24h",
) -> PredictionResponse:
    """Forecast range of the price at the given horizon."""
    return service.predict(symbol, horizon)
