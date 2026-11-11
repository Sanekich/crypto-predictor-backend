from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Path as ApiPath

from app.api.dependencies import _resolve
from app.core.config import get_settings

router = APIRouter(prefix="/training", tags=["training"])

FEATURE_STAGES = [
    (1, "v1 — OHLCV", "Текущая базовая модель", []),
    (2, "v2 — + funding", "Добавить funding rate", ["mark_price"]),
    (3, "v3 — + open interest", "Добавить open interest", ["open_interest"]),
    (4, "v4 — + liquidations", "Добавить ликвидации", ["liquidations"]),
    (5, "v5 — + trade flow", "Добавить поток сделок", ["trades_spot", "trades_futures"]),
    (6, "v6 — + order book", "Добавить дисбаланс стакана", ["book_spot", "book_futures"]),
]


def _coverage_hours(root: Path, symbol: str, datasets: list[str]) -> tuple[int, int]:
    hours: set[str] = set()
    files = 0
    for dataset in datasets:
        for file in (root / dataset / symbol.upper()).glob("*/*.parquet"):
            files += 1
            hours.add(f"{file.parent.name}/{file.stem}")
    return files, len(hours)


def _latest_model_stage(artifacts_root: Path, symbol: str) -> int:
    latest_stage = 1
    for metadata_path in artifacts_root.glob(f"*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if metadata.get("symbol") != symbol.upper():
            continue
        columns = " ".join(metadata.get("feature_columns", []))
        for stage, _, _, datasets in FEATURE_STAGES[1:]:
            if any(dataset in columns.lower() for dataset in datasets):
                latest_stage = max(latest_stage, stage)
    return latest_stage


@router.get("/stage/{symbol}")
def training_stage(symbol: str = ApiPath(pattern=r"^[A-Za-z0-9]{2,20}$")) -> dict[str, Any]:
    settings = get_settings()
    live_root = _resolve(settings.data_dir) / "live"
    artifacts_root = _resolve(settings.model_artifacts_dir)
    current_stage = _latest_model_stage(artifacts_root, symbol)
    next_stage = min(current_stage + 1, len(FEATURE_STAGES))
    stages: list[dict[str, Any]] = []
    for stage, label, description, datasets in FEATURE_STAGES:
        files, coverage_hours = _coverage_hours(live_root, symbol, datasets) if datasets else (0, 0)
        ready = not datasets or coverage_hours >= 24 * 7
        if stage < current_stage:
            state = "completed"
        elif stage == current_stage:
            state = "current"
        elif ready:
            state = "ready_to_test"
        else:
            state = "collecting"
        stages.append({
            "stage": stage,
            "label": label,
            "description": description,
            "datasets": datasets,
            "files": files,
            "coverage_hours": coverage_hours,
            "ready": ready,
            "state": state,
        })
    next_info = stages[next_stage - 1]
    if current_stage == len(FEATURE_STAGES):
        recommendation = "Все запланированные этапы пройдены. Сравни последнюю модель с baseline и не меняй production без shadow-проверки."
    elif next_info["ready"]:
        recommendation = f"Можно начинать shadow-эксперимент {next_info['label']}. Сначала сравни его с текущей моделью через walk-forward."
    else:
        remaining = max(0, 24 * 7 - next_info["coverage_hours"])
        recommendation = f"Пока не переходи. Собери ещё примерно {remaining} часов данных для этапа {next_info['label']}, затем переобучи shadow-модель."
    return {
        "symbol": symbol.upper(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_stage": current_stage,
        "current_label": stages[current_stage - 1]["label"],
        "next_stage": None if current_stage == len(FEATURE_STAGES) else next_stage,
        "next_ready": bool(next_info["ready"]) if current_stage < len(FEATURE_STAGES) else False,
        "recommendation": recommendation,
        "stages": stages,
    }
