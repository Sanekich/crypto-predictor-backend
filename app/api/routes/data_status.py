import json
from typing import Any

from fastapi import APIRouter, Path, Query

from app.api.dependencies import _resolve
from app.core.config import get_settings
from app.core.exceptions import NotFoundError

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/status/{symbol}")
def data_status(
    symbol: str = Path(pattern=r"^[A-Za-z0-9]{2,20}$"),
    interval: str = Query(default="1h", pattern=r"^\d+[mhdw]$"),
) -> dict[str, Any]:
    """Quality and coverage summary for the processed OHLCV dataset."""
    settings = get_settings()
    path = _resolve(settings.data_dir) / "processed" / settings.market_data_provider / f"{symbol.upper()}_{interval}.report.json"
    if not path.exists():
        raise NotFoundError(f"No processed data report for {symbol.upper()} {interval}.")
    return json.loads(path.read_text(encoding="utf-8"))
