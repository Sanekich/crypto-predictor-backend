from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import get_signal_service
from app.schemas.signals import SignalsResponse
from app.services.signal_service import DEFAULT_THRESHOLD, SignalService

router = APIRouter(tags=["signals"])


@router.get("/signals/{symbol}", response_model=SignalsResponse)
def get_signals(
    service: Annotated[SignalService, Depends(get_signal_service)],
    symbol: Annotated[str, Path(pattern=r"^[A-Za-z0-9]{2,20}$", examples=["BTCUSDT"])],
    min_confidence: Annotated[float, Query(ge=0.5, le=1.0)] = DEFAULT_THRESHOLD,
) -> SignalsResponse:
    """Short-horizon (1-4 h) direction probabilities, high-confidence signals and their track record."""
    return service.get_signals(symbol, min_confidence)
