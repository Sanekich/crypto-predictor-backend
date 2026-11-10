import json
from typing import Any

from fastapi import APIRouter, Path

from app.api.dependencies import _resolve
from app.core.config import get_settings
from app.core.exceptions import NotFoundError

router = APIRouter(prefix="/live", tags=["live data"])


@router.get("/status/{symbol}")
def live_status(symbol: str = Path(pattern=r"^[A-Za-z0-9]{2,20}$")) -> dict[str, Any]:
    """State of the real-time collector (scripts/collect_live.py): connections, messages, rows written."""
    path = _resolve(get_settings().data_dir) / "live" / f"_status_{symbol.upper()}.json"
    if not path.exists():
        raise NotFoundError(f"No live collector status for {symbol.upper()}. Is scripts/collect_live.py running?")
    return json.loads(path.read_text(encoding="utf-8"))
