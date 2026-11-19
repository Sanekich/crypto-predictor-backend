"""Paper-trading only: independent virtual wallets for current models."""
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
STARTING_BALANCE = 10000.0


def _path() -> Path:
    return _resolve(get_settings().data_dir) / "paper_trading" / "portfolio.json"


def _empty() -> dict[str, Any]:
    return {"wallets": {}}


def _load() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return _empty()
    data = json.loads(path.read_text(encoding="utf-8"))
    if "wallets" in data:
        return data
    # One-time migration from the previous shared-wallet format.
    wallets: dict[str, Any] = {}
    for trade in data.get("trades", []):
        wallets.setdefault(trade["model_id"], {"initial_balance": STARTING_BALANCE, "cash": STARTING_BALANCE, "position": None, "trades": []})["trades"].append(trade)
    for model_id, position in data.get("positions", {}).items():
        wallets.setdefault(model_id, {"initial_balance": STARTING_BALANCE, "cash": STARTING_BALANCE, "position": None, "trades": []})["position"] = position
    return {"wallets": wallets}


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


def _wallet(data: dict[str, Any], model_id: str) -> dict[str, Any]:
    return data["wallets"].setdefault(model_id, {"initial_balance": STARTING_BALANCE, "cash": STARTING_BALANCE, "position": None, "trades": []})


def _wallet_view(wallet: dict[str, Any], current_price: float) -> dict[str, Any]:
    position = wallet.get("position")
    unrealized = 0.0
    view_position = None
    if position:
        move = current_price / position["entry_price"] - 1.0
        unrealized = position["notional"] * (move if position["side"] == "long" else -move)
        view_position = {**position, "current_price": current_price, "unrealized_pnl": unrealized}
    realized = sum(float(trade.get("pnl", 0.0)) for trade in wallet.get("trades", []))
    equity = float(wallet["cash"]) + ((position["notional"] + unrealized) if position else 0.0)
    return {"initial_balance": wallet["initial_balance"], "cash": wallet["cash"], "equity": equity, "realized_pnl": realized, "unrealized_pnl": unrealized, "return_pct": equity / wallet["initial_balance"] - 1.0, "position": view_position, "trades": wallet.get("trades", [])[-10:]}


def _overview(data: dict[str, Any], current_price: float, models: list[dict[str, Any]]) -> dict[str, Any]:
    model_views = []
    for model in models:
        model_views.append({**model, "wallet": _wallet_view(_wallet(data, model["model_id"]), current_price)})
    if models:
        _save(data)
    equity = sum(model["wallet"]["equity"] for model in model_views)
    initial = sum(model["wallet"]["initial_balance"] for model in model_views)
    cash = sum(model["wallet"]["cash"] for model in model_views)
    return {"initial_balance": initial, "cash": cash, "equity": equity, "realized_pnl": sum(m["wallet"]["realized_pnl"] for m in model_views), "unrealized_pnl": sum(m["wallet"]["unrealized_pnl"] for m in model_views), "return_pct": equity / initial - 1.0 if initial else 0.0, "current_price": current_price, "models": model_views, "virtual_only": True}


class TradeRequest(BaseModel):
    model_id: str = Field(min_length=1)
    side: Literal["long", "short"]
    notional: float = Field(gt=10, le=10000)


@router.get("/overview")
def overview(registry: Registry, barrier: BarrierRegistry, experiments: ExperimentRegistry, market: MarketData) -> dict[str, Any]:
    with _lock:
        return _overview(_load(), _price(market), _models(registry, barrier, experiments))


@router.post("/trade")
def open_trade(request: TradeRequest, registry: Registry, barrier: BarrierRegistry, experiments: ExperimentRegistry, market: MarketData, x_operations_token: str | None = Header(default=None), x_operations_token_b64: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(x_operations_token, x_operations_token_b64)
    with _lock:
        data = _load()
        wallet = _wallet(data, request.model_id)
        if wallet.get("position"):
            raise HTTPException(status_code=409, detail="У этой модели уже есть открытая виртуальная позиция")
        if request.notional > wallet["cash"]:
            raise HTTPException(status_code=409, detail="Недостаточно виртуального баланса этой модели")
        price = _price(market)
        wallet["cash"] -= request.notional
        wallet["position"] = {"side": request.side, "notional": request.notional, "entry_price": price, "opened_at": datetime.now(timezone.utc).isoformat()}
        _save(data)
        return _overview(data, price, _models(registry, barrier, experiments))


@router.post("/close/{model_id}")
def close_trade(model_id: str, registry: Registry, barrier: BarrierRegistry, experiments: ExperimentRegistry, market: MarketData, x_operations_token: str | None = Header(default=None), x_operations_token_b64: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(x_operations_token, x_operations_token_b64)
    with _lock:
        data = _load()
        wallet = _wallet(data, model_id)
        position = wallet.get("position")
        if position is None:
            raise HTTPException(status_code=404, detail="Позиция не найдена")
        price = _price(market)
        move = price / position["entry_price"] - 1.0
        pnl = position["notional"] * (move if position["side"] == "long" else -move)
        wallet["cash"] += position["notional"] + pnl
        wallet["trades"].append({"model_id": model_id, **position, "exit_price": price, "pnl": pnl, "closed_at": datetime.now(timezone.utc).isoformat()})
        wallet["position"] = None
        _save(data)
        return _overview(data, price, _models(registry, barrier, experiments))


@router.post("/reset")
def reset_portfolio(x_operations_token: str | None = Header(default=None), x_operations_token_b64: str | None = Header(default=None)) -> dict[str, str]:
    _check_token(x_operations_token, x_operations_token_b64)
    with _lock:
        _save(_empty())
    return {"status": "reset", "balance_per_model": f"{STARTING_BALANCE:.2f}"}
