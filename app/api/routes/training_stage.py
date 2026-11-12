"""Readiness of the feature stages from docs/direction_prediction_plan.md.

Stages 2-3 use downloadable history (scripts/download_derivatives.py) and can be
tested right away. Stages 4-6 need data recorded by the live collector; they are
considered ready only after LIVE_REQUIRED_HOURS of coverage - with less data a
model would learn a single market regime.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Path as ApiPath

from app.api.dependencies import _resolve
from app.core.config import get_settings

router = APIRouter(prefix="/training", tags=["training"])

LIVE_REQUIRED_HOURS = 24 * 60  # about two months

# (stage, label, description, source kind, datasets, derivatives groups used by the stage)
FEATURE_STAGES: list[tuple[int, str, str, str, list[str], tuple[str, ...]]] = [
    (1, "v1 — OHLCV", "Свечи: доходности, индикаторы, волатильность, объём", "history", [], ()),
    (2, "v2 — + funding и basis", "Funding rate и разница фьючерса и спота (история с 2019)", "history", ["funding", "premium"], ("funding", "basis")),
    (3, "v3 — + open interest и DVOL", "Открытый интерес, позиции трейдеров, индекс волатильности (история с 2020–2021)", "history", ["metrics", "dvol"], ("open_interest", "dvol")),
    (4, "v4 — + ликвидации", "Ликвидации (только живые данные)", "live", ["liquidations"], ()),
    (5, "v5 — + поток сделок", "Агрессивные покупки и продажи (только живые данные)", "live", ["trades_spot", "trades_futures"], ()),
    (6, "v6 — + стакан", "Дисбаланс заявок (только живые данные)", "live", ["book_spot", "book_futures"], ()),
]


def _live_coverage(root: Path, symbol: str, datasets: list[str]) -> tuple[int, int]:
    """Files and hours covered by ALL datasets of the stage."""
    files = 0
    common: set[str] | None = None
    for dataset in datasets:
        hours = set()
        for file in (root / dataset / symbol.upper()).glob("*/*.parquet"):
            files += 1
            hours.add(f"{file.parent.name}/{file.stem}")
        common = hours if common is None else common & hours
    return files, len(common or set())


def _history_ready(derivatives_root: Path, symbol: str, datasets: list[str]) -> tuple[int, bool]:
    paths = [derivatives_root / d / f"{symbol.upper()}.parquet" for d in datasets]
    present = [p for p in paths if p.exists()]
    return len(present), len(present) == len(paths)


def _model_stage(feature_config: dict[str, Any]) -> int:
    groups = set(feature_config.get("derivatives") or [])
    stage = 1
    for number, *_rest, stage_groups in FEATURE_STAGES:
        if stage_groups and groups & set(stage_groups):
            stage = max(stage, number)
    return stage


def _latest_model_stage(artifacts_root: Path, symbol: str) -> int:
    latest_stage = 1
    for metadata_path in (artifacts_root / "models").glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if metadata.get("symbol") == symbol.upper():
            latest_stage = max(latest_stage, _model_stage(metadata.get("feature_config") or {}))
    return latest_stage


@router.get("/stage/{symbol}")
def training_stage(symbol: str = ApiPath(pattern=r"^[A-Za-z0-9]{2,20}$")) -> dict[str, Any]:
    settings = get_settings()
    data_root = _resolve(settings.data_dir)
    current_stage = _latest_model_stage(_resolve(settings.model_artifacts_dir), symbol)
    next_stage = min(current_stage + 1, len(FEATURE_STAGES))

    stages: list[dict[str, Any]] = []
    for stage, label, description, kind, datasets, _groups in FEATURE_STAGES:
        if kind == "live":
            files, coverage_hours = _live_coverage(data_root / "live", symbol, datasets)
            ready = coverage_hours >= LIVE_REQUIRED_HOURS
            required_hours = LIVE_REQUIRED_HOURS
        elif datasets:
            files, ready = _history_ready(data_root / "derivatives", symbol, datasets)
            coverage_hours, required_hours = 0, 0
        else:
            files, ready, coverage_hours, required_hours = 0, True, 0, 0

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
            "source": kind,
            "datasets": datasets,
            "files": files,
            "coverage_hours": coverage_hours,
            "required_hours": required_hours,
            "ready": ready,
            "state": state,
        })

    next_info = stages[next_stage - 1]
    if current_stage == len(FEATURE_STAGES):
        recommendation = "Все этапы пройдены. Новая модель заменяет старую только после walk-forward и теневого режима."
    elif next_info["ready"] and next_info["source"] == "history":
        recommendation = (
            f"Данные для этапа {next_info['label']} скачаны. Сравните наборы признаков: "
            "python scripts/compare_feature_sets.py — переходить, только если вердикт BETTER."
        )
    elif next_info["source"] == "history":
        recommendation = f"Скачайте историю для этапа {next_info['label']}: python scripts/download_derivatives.py"
    elif next_info["ready"]:
        recommendation = f"Живых данных достаточно для этапа {next_info['label']}. Сначала walk-forward, затем теневой режим."
    else:
        remaining_days = max(0, next_info["required_hours"] - next_info["coverage_hours"]) / 24
        recommendation = (
            f"Этап {next_info['label']}: собрано {next_info['coverage_hours'] / 24:.0f} из "
            f"{next_info['required_hours'] / 24:.0f} дней живых данных, осталось ~{remaining_days:.0f}. "
            "Не выключайте сборщик."
        )
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
