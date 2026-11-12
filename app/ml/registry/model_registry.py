"""Local registry of trained models.

Each model version is a directory:
    artifacts/models/<model_id>/
        model.json | model.joblib   model weights
        params.json                 (xgboost) training parameters
        metadata.json               everything needed to reproduce and use the model

A model must always be loaded together with its metadata: the feature list,
their order and the feature pipeline version define what input it expects.
"""

import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ml.models.base import ReturnModel
from app.ml.models.baselines import HistoricalMeanBaseline, RidgeBaseline, ZeroReturnBaseline
from app.ml.models.direction_model import XGBoostDirectionModel
from app.ml.models.neutral_direction_model import XGBoostNeutralDirectionModel
from app.ml.models.quantile_model import HistoricalQuantileBaseline, XGBoostQuantileModel
from app.ml.models.xgboost_model import XGBoostReturnModel

MODEL_TYPES: dict[str, type[ReturnModel]] = {
    cls.name: cls
    for cls in (
        XGBoostReturnModel,
        RidgeBaseline,
        HistoricalMeanBaseline,
        ZeroReturnBaseline,
        XGBoostQuantileModel,
        HistoricalQuantileBaseline,
        XGBoostDirectionModel,
        XGBoostNeutralDirectionModel,
    )
}

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def make_model_id(model_type: str, symbol: str, interval: str, horizon: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"{model_type}-{symbol.lower()}-{interval}-h{horizon}-{stamp}"


def _json_safe(value: Any) -> Any:
    """NaN/inf are not valid JSON: store them as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class ModelNotFoundError(LookupError):
    pass


class ModelRegistry:
    def __init__(self, artifacts_dir: Path | str) -> None:
        self.models_dir = Path(artifacts_dir) / "models"

    def _dir(self, model_id: str) -> Path:
        if not _SAFE_ID.match(model_id):
            raise ValueError(f"Invalid model id: {model_id!r}")
        return self.models_dir / model_id

    def save(self, model: ReturnModel, model_id: str, metadata: dict[str, Any]) -> Path:
        directory = self._dir(model_id)
        if directory.exists():
            raise FileExistsError(f"Model {model_id} already exists")
        model.save(directory)
        full_metadata = {"model_id": model_id, "model_type": model.name, **metadata}
        tmp = directory / "metadata.json.tmp"
        tmp.write_text(json.dumps(_json_safe(full_metadata), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, directory / "metadata.json")
        return directory

    def metadata(self, model_id: str) -> dict[str, Any]:
        path = self._dir(model_id) / "metadata.json"
        if not path.exists():
            raise ModelNotFoundError(model_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, model_id: str) -> tuple[ReturnModel, dict[str, Any]]:
        meta = self.metadata(model_id)
        model_cls = MODEL_TYPES.get(meta["model_type"])
        if model_cls is None:
            raise ValueError(f"Unknown model type: {meta['model_type']}")
        return model_cls.load(self._dir(model_id)), meta

    def list(self) -> list[dict[str, Any]]:
        """Metadata of all saved models, newest first."""
        if not self.models_dir.exists():
            return []
        items = []
        for directory in self.models_dir.iterdir():
            if (directory / "metadata.json").exists():
                items.append(self.metadata(directory.name))
        items = sorted(items, key=lambda m: m.get("trained_at", ""), reverse=True)
        latest_keys: set[tuple[Any, ...]] = set()
        latest_ids: set[str] = set()
        for meta in items:
            key = (meta.get("model_type"), meta.get("symbol"), meta.get("interval"), meta.get("horizon"))
            if key not in latest_keys:
                latest_keys.add(key)
                latest_ids.add(meta["model_id"])
        return [{**meta, "is_latest": meta.get("model_id") in latest_ids} for meta in items]

    def delete_old(self, model_id: str) -> None:
        """Delete a historical artifact, never the latest or an approved model."""
        meta = self.metadata(model_id)
        if meta.get("status") == "approved":
            raise ValueError("Approved models cannot be deleted")
        if self.latest(meta.get("model_type", ""), meta.get("symbol", ""), meta.get("interval", ""), meta.get("horizon", "")):
            latest = self.latest(meta["model_type"], meta["symbol"], meta["interval"], meta["horizon"])
            if latest and latest.get("model_id") == model_id:
                raise ValueError("The latest model version cannot be deleted")
        directory = self._dir(model_id)
        shutil.rmtree(directory)

    def latest(self, model_type: str, symbol: str, interval: str, horizon: str) -> dict[str, Any] | None:
        """Metadata of the most recently trained matching model, or None."""
        for meta in self.list():
            if (
                meta.get("model_type") == model_type
                and meta.get("symbol") == symbol.upper()
                and meta.get("interval") == interval
                and meta.get("horizon") == horizon
            ):
                return meta
        return None
