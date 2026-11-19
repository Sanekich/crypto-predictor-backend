"""Paper-trading only: virtual portfolio, current model cards, and simulated positions."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import _resolve, get_barrier_registry, get_experiment_registry, get_market_data_service, get_registry
from app.api.routes.operations import _check_token
from app.core.config import get_settings
from app.ml.registry.model_registry import ModelRegistry
from app.services.market_data_service import MarketDataService

router = APIRouter(prefix="/paper-trading", tags=["paper-trading"])
Registry = Annotated[ModelRegistry, Depends(get_registry)]
BarrierRegistry = Annotated[ModelRegistry, Depends(get_barrier_registry)]
ExperimentRegistry = Annotated[ModelRegistry, Depends(get_experiment_registry)]
MarketData = Annotated[MarketDataService, Depends(get_market_data_service)]
_lock = threading.Lock()


def _path() -> Path:
    return _resolve(get_settings().data_dir) / "paper_trading" / "portfolio.json"


def _load() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"initial_balance": 10000.0, "cash": 10000.0, "positions": {}, "trades": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(value: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _price(market: MarketData) -> float:
    return float(market.get_ohlcv("BTCUSDT", "1h").iloc[-1]["close"])


def _models(registry: ModelRegistry, barrier: ModelRegistry, experiments: ModelRegistry) -> list[dict[str, Any]]:
    all_meta = registry.list() + barrier.list() + experiments.list()
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for meta in all_meta:
        key = (meta.get("model_type", "unknown"), meta.get("symbol", "BTCUSDT"), meta.get("interval", "1h"), meta.get("horizon", "24h"))
        if key not in latest or meta.get("trained_at", "") > latest[key].get("trained_at", ""):
            latest[key] = meta
    result = []
    for meta in latest.values():
        result.append({
            "model_id": meta.get("model_id"), "model_type": meta.get("model_type"), "status": meta.get("status", "shadow"),
            "trained_at": meta.get("trained_at"), "symbol": meta.get("symbol", "BTCUSDT"), "interval": meta.get("interval", "1h"),
            "horizon": meta.get("horizon", "24h"), "feature_set_label": meta.get("feature_set_label", meta.get("feature_set", "")),
            "logic": meta.get("logic", "research"), "paper_enabled": True,
        })
    return sorted(result, key=lambda row: (row["status"] != "approved", row["model_type"], row["trained_at"] or ""))


def _overview(data: dict[str, Any], current_price: float, models: list[dict[str, Any]]) -> dict[str, Any]:
    positions = []
    unrealized = 0.0
    for model_id, position in data["positions"].items():
        move = current_price / position["entry_price"] - 1.0
        pnl = position["notional"] * (move if position["side"] == "long" else -move)
        unrealized += pnl
        positions.append({**position, "model_id": model_id, "current_price": current_price, "unrealized_pnl": pnl})
    realized = sum(float(trade.get("pnl", 0.0)) for trade in data["trades"])
    equity = float(data["cash"]) + sum(p["notional"] + p["unrealized_pnl"] for p in positions)
    return {"initial_balance": data["initial_balance"], "cash": data["cash"], "equity": equity, "realized_pnl": realized, "unrealized_pnl": unrealized, "return_pct": (equity / data["initial_balance"] - 1.0) if data["initial_balance"] else 0.0, "current_price": current_price, "positions": positions, "trades": data["trades"][-30:], "models": models, "virtual_only": True}


class TradeRequest(BaseModel):
    model_id: str = Field(min_length=1)
    side: Literal["long", "short"]
    notional: float = Field(gt=10, le=10000)


@router.get("/overview")
def overview(registry: Registry, barrier: BarrierRegistry, experiments: ExperimentRegistry, market: MarketData) -> dict[str, Any]:
    with _lock:
        return _overview(_load(), _price(market), _models(registry, barrier, experiments))


@router.post("/trade")
def open_trade(
    request: TradeRequest,
    registry: Registry,
    barrier: BarrierRegistry,
    experiments: ExperimentRegistry,
    market: MarketData,
    x_operations_token: str | None = Header(default=None),
    x_operations_token_b64: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_token(x_operations_token, x_operations_token_b64)
    with _lock:
        data = _load()
        if request.model_id in data["positions"]:
            raise HTTPException(status_code=409, detail="У этой модели уже есть открытая виртуальная позиция")
        if request.notional > data["cash"]:
            raise HTTPException(status_code=409, detail="Недостаточно виртуального баланса")
        price = _price(market)
        data["cash"] -= request.notional
        data["positions"][request.model_id] = {"side": request.side, "notional": request.notional, "entry_price": price, "opened_at": datetime.now(timezone.utc).isoformat()}
        _save(data)
        return _overview(data, price, _models(registry, barrier, experiments))


@router.post("/close/{model_id}")
def close_trade(
    model_id: str,
    registry: Registry,
    barrier: BarrierRegistry,
    experiments: ExperimentRegistry,
    market: MarketData,
    x_operations_token: str | None = Header(default=None),
    x_operations_token_b64: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_token(x_operations_token, x_operations_token_b64)
    with _lock:
        data = _load()
        position = data["positions"].pop(model_id, None)
        if position is None:
            raise HTTPException(status_code=404, detail="Позиция не найдена")
        price = _price(market)
        move = price / position["entry_price"] - 1.0
        pnl = position["notional"] * (move if position["side"] == "long" else -move)
        data["cash"] += position["notional"] + pnl
        data["trades"].append({"model_id": model_id, **position, "exit_price": price, "pnl": pnl, "closed_at": datetime.now(timezone.utc).isoformat()})
        _save(data)
        return _overview(data, price, _models(registry, barrier, experiments))


@router.post("/reset")
def reset_portfolio(
    x_operations_token: str | None = Header(default=None),
    x_operations_token_b64: str | None = Header(default=None),
) -> dict[str, str]:
    _check_token(x_operations_token, x_operations_token_b64)
    with _lock:
        _save({"initial_balance": 10000.0, "cash": 10000.0, "positions": {}, "trades": []})
    return {"status": "reset", "balance": "10000.00"}
