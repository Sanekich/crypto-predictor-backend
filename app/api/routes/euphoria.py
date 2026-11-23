"""Euphoria / panic index: current value, recent history and the latest event study."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter

from app.api.dependencies import _resolve
from app.core.config import get_settings
from app.data.derivatives.store import DerivativesStore
from app.data.storage import MarketDataStorage
from app.ml.euphoria.index import COMPONENT_LABELS, COMPONENTS, REQUIRED_DATASETS, build_euphoria_frame
from app.ml.euphoria.study import GATE, best_results

router = APIRouter(prefix="/euphoria", tags=["euphoria"])

ZONES = ((2.0, "сильная эйфория"), (1.5, "эйфория"), (-1.5, "спокойно"), (-2.0, "паника"))
_cache: dict[str, Any] = {"key": None, "frame": None}
_lock = threading.Lock()


def _zone(value: float) -> str:
    if value >= 2.0:
        return "strong_euphoria"
    if value >= 1.5:
        return "euphoria"
    if value <= -2.0:
        return "strong_panic"
    if value <= -1.5:
        return "panic"
    return "neutral"


def _frame(data_dir: Path) -> pd.DataFrame | None:
    """Index frame, rebuilt only when the underlying files change (it takes a few seconds)."""
    store = DerivativesStore(data_dir)
    storage = MarketDataStorage(data_dir)
    candles_path = data_dir / "processed" / "binance" / "BTCUSDT_1h.parquet"
    paths = [candles_path, *(store.path(name, "BTCUSDT") for name in REQUIRED_DATASETS)]
    if not all(p.exists() for p in paths):
        return None
    key = tuple(p.stat().st_mtime_ns for p in paths)
    with _lock:
        if _cache["key"] != key:
            ohlcv = storage.load_processed(get_settings().market_data_provider, "BTCUSDT", "1h")
            _cache["frame"] = build_euphoria_frame(ohlcv, store.load_all("BTCUSDT"))
            _cache["key"] = key
        return _cache["frame"]


def _latest_report(data_dir: Path) -> dict[str, Any] | None:
    reports = sorted((data_dir / "reports").glob("euphoria_*.json"))
    return json.loads(reports[-1].read_text(encoding="utf-8")) if reports else None


@router.get("/overview")
def overview() -> dict[str, Any]:
    data_dir = _resolve(get_settings().data_dir)
    frame = _frame(data_dir)
    current = None
    history: list[dict[str, Any]] = []
    if frame is not None:
        valid = frame.dropna(subset=["euphoria"])
        if len(valid):
            row = valid.iloc[-1]
            as_of = row["timestamp"] + pd.Timedelta(hours=1)
            current = {
                "as_of": as_of.isoformat(),
                "age_hours": round((datetime.now(timezone.utc) - as_of.to_pydatetime()).total_seconds() / 3600, 1),
                "value": float(row["euphoria"]),
                "zone": _zone(float(row["euphoria"])),
                "price": float(row["close"]),
                "components": [
                    {"key": name, "label": COMPONENT_LABELS[name], "raw": float(row[name]), "z": float(row[f"z_{name}"])}
                    for name in COMPONENTS
                ],
            }
            recent = valid[valid["timestamp"] >= valid["timestamp"].iloc[-1] - pd.Timedelta(days=365)]
            daily = recent.set_index("timestamp")[["euphoria", "close"]].resample("1D").last().dropna()
            history = [{"date": ts.date().isoformat(), "value": float(v.euphoria), "price": float(v.close)} for ts, v in daily.iterrows()]
    report = _latest_report(data_dir)
    return {
        "question": "Насколько толпа перекошена в одну сторону?",
        "thresholds": {"euphoria": 1.5, "strong": 2.0},
        "components": [{"key": k, "label": COMPONENT_LABELS[k]} for k in COMPONENTS],
        "current": current,
        "history": history,
        "study": None if report is None else {
            "created_at": report.get("created_at"),
            "period": report.get("period"),
            "gate": report.get("gate", GATE),
            "results": [{k: v for k, v in r.items() if k != "by_year"} for r in report["results"]],
            "best": best_results(report),
        },
    }
