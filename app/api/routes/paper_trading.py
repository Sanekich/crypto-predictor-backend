"""Paper trading (virtual only): strategy forward test + manual wallets per model.

The strategy engine (app/services/paper_strategies.py) runs in the background every
minute (see app/main.py) and trades fixed rules with fees and horizon exits. The
manual wallets let a person open/close virtual positions for any current model.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import (
    _resolve,
    get_barrier_registry,
    get_experiment_registry,
    get_market_data_service,
    get_registry,
    get_signal_service,
    get_volatility_registry,
)
from app.api.routes.operations import _check_token
from app.core.config import get_settings
from app.data.derivatives.store import DerivativesStore
from app.features.feature_pipeline import FeatureConfig, build_features, feature_columns
from app.ml.experiments.state_features import build_state_features
from app.ml.registry.model_registry import ModelRegistry
from app.services.market_data_service import MarketDataService
from app.services.paper_strategies import STRATEGIES, StrategyInputs, new_wallet, strategy_view, tick, utc_now
from app.services.signal_service import SignalService

router = APIRouter(prefix="/paper-trading", tags=["paper-trading"])
Registry = Annotated[ModelRegistry, Depends(get_registry)]
BarrierRegistry = Annotated[ModelRegistry, Depends(get_barrier_registry)]
ExperimentRegistry = Annotated[ModelRegistry, Depends(get_experiment_registry)]
MarketData = Annotated[MarketDataService, Depends(get_market_data_service)]
Signal = Annotated[SignalService, Depends(get_signal_service)]
_lock = threading.Lock()
logger = logging.getLogger(__name__)
STARTING_BALANCE = 10000.0


def _path() -> Path:
    return _resolve(get_settings().data_dir) / "paper_trading" / "portfolio.json"


def _empty() -> dict[str, Any]:
    return {"wallets": {}, "strategies": {}, "controller": {"enabled": True, "last_tick": None, "last_action": "waiting for the first tick", "threshold": 0.60}}


def _load() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return _empty()
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("controller", {"enabled": True, "last_tick": None, "last_action": "", "threshold": 0.60})
    data.setdefault("strategies", {})
    if "wallets" in data:
        return data
    # One-time migration from the previous shared-wallet format.
    wallets: dict[str, Any] = {}
    for trade in data.get("trades", []):
        wallets.setdefault(trade["model_id"], {"initial_balance": STARTING_BALANCE, "cash": STARTING_BALANCE, "position": None, "trades": []})["trades"].append(trade)
    for model_id, position in data.get("positions", {}).items():
        wallets.setdefault(model_id, {"initial_balance": STARTING_BALANCE, "cash": STARTING_BALANCE, "position": None, "trades": []})["position"] = position
    return {"wallets": wallets, "controller": {"enabled": False, "last_tick": None, "last_action": "off", "threshold": 0.60}}


def _save(value: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _price(market: MarketData) -> float:
    return float(market.get_ohlcv("BTCUSDT", "1h").iloc[-1]["close"])


def _state_forecasts(experiments: ModelRegistry, market: MarketData, models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Infer only from saved state-aware artifacts; no fallback or guessed signal."""
    result: dict[str, dict[str, Any]] = {}
    candles = market.get_ohlcv("BTCUSDT", "1h")
    data_dir = _resolve(get_settings().data_dir)
    for card in models:
        if card.get("model_type") != "xgboost_state_aware_barrier" or not card.get("promising"):
            continue
        try:
            meta = experiments.metadata(card["model_id"])
            config = FeatureConfig.from_dict(meta.get("feature_config") or {})
            derivatives = None
            if meta.get("feature_set") == "all_data":
                derivatives = DerivativesStore(data_dir).load_all("BTCUSDT")
            base = build_features(candles, "1h", config, derivatives)
            state = build_state_features(base)
            frame = base.join(state)
            columns = meta.get("feature_columns") or [*feature_columns(base), *state.columns]
            row = frame.dropna(subset=columns).iloc[-1]
            model, _ = experiments.load(card["model_id"])
            p_up = float(model.predict_proba_up(row[columns].to_numpy(dtype="float64").reshape(1, -1))[0])
            confidence = max(p_up, 1.0 - p_up)
            result[card["model_id"]] = {"horizon": meta.get("horizon", "24h"), "direction": "up" if p_up >= 0.5 else "down", "p_up": p_up, "confidence": confidence, "is_signal": bool(confidence >= 0.60 and meta.get("walk_forward", {}).get("has_skill")), "has_skill": bool(meta.get("walk_forward", {}).get("has_skill")), "target_time": row["timestamp"].isoformat(), "status": meta.get("status", "shadow")}
        except Exception:
            continue
    return result


def _barrier_forecasts(barrier: ModelRegistry, market: MarketData, models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Run live inference for saved regular XGBoost barrier artifacts."""
    result: dict[str, dict[str, Any]] = {}
    candles = market.get_ohlcv("BTCUSDT", "1h")
    data_dir = _resolve(get_settings().data_dir)
    for card in models:
        if card.get("model_type") != "xgboost_barrier_direction" or not card.get("promising"):
            continue
        try:
            meta = barrier.metadata(card["model_id"])
            config = FeatureConfig.from_dict(meta.get("feature_config") or {})
            derivatives = DerivativesStore(data_dir).load_all("BTCUSDT") if meta.get("feature_set") == "all_data" else None
            base = build_features(candles, meta.get("interval", "1h"), config, derivatives)
            columns = meta.get("feature_columns") or feature_columns(base)
            row = base.dropna(subset=columns).iloc[-1]
            model, _ = barrier.load(card["model_id"])
            p_up = float(model.predict_proba_up(row[columns].to_numpy(dtype="float64").reshape(1, -1))[0])
            confidence = max(p_up, 1.0 - p_up)
            result[card["model_id"]] = {"horizon": meta.get("horizon", "24h"), "direction": "up" if p_up >= 0.5 else "down", "p_up": p_up, "confidence": confidence, "is_signal": bool(confidence >= 0.60 and meta.get("walk_forward", {}).get("has_skill")), "has_skill": bool(meta.get("walk_forward", {}).get("has_skill")), "target_time": row["timestamp"].isoformat(), "status": meta.get("status", "shadow")}
        except Exception:
            continue
    return result


def _models(registry: ModelRegistry, barrier: ModelRegistry, experiments: ModelRegistry) -> list[dict[str, Any]]:
    all_meta = registry.list() + barrier.list() + experiments.list()
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for meta in all_meta:
        key = (meta.get("model_type", "unknown"), meta.get("symbol", "BTCUSDT"), meta.get("interval", "1h"), meta.get("horizon", "24h"))
        if key not in latest or meta.get("trained_at", "") > latest[key].get("trained_at", ""):
            latest[key] = meta
    result = []
    for meta in latest.values():
        walk = meta.get("walk_forward") or {}
        test = (meta.get("metrics") or {}).get("test") or {}
        positive = walk.get("positive_folds")
        folds = walk.get("folds")
        skill = walk.get("brier_skill_mean")
        if skill is None:
            skill = walk.get("mean_brier_skill")
        reasons: list[str] = []
        if meta.get("status") == "approved":
            reasons.append("approved")
        if positive is not None and folds and positive / folds >= 0.6:
            reasons.append(f"{positive}/{folds} positive folds")
        if skill is not None and skill > 0:
            reasons.append("positive walk-forward skill")
        if walk.get("has_skill"):
            reasons.append("stable skill gate passed")
        if test.get("accepted_accuracy") is not None and test.get("primary_accuracy") is not None and test["accepted_accuracy"] > test["primary_accuracy"]:
            reasons.append("meta filter improves accepted accuracy")
        promising = bool(meta.get("status") == "approved" or walk.get("has_skill") or (positive is not None and folds and positive / folds >= 0.6 and (skill is None or skill >= 0)))
        result.append({
            "model_id": meta.get("model_id"), "model_type": meta.get("model_type"), "status": meta.get("status", "shadow"),
            "trained_at": meta.get("trained_at"), "symbol": meta.get("symbol", "BTCUSDT"), "interval": meta.get("interval", "1h"),
            "horizon": meta.get("horizon", "24h"), "feature_set_label": meta.get("feature_set_label", meta.get("feature_set", "")),
            "logic": meta.get("logic", "research"), "paper_enabled": True,
            "promising": promising, "promise_reason": ", ".join(reasons) if reasons else "нет устойчивого преимущества пока",
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


def _overview(data: dict[str, Any], current_price: float, models: list[dict[str, Any]], forecasts: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    model_views = []
    for model in models:
        model_views.append({**model, "forecast": forecasts.get(model["model_id"]) if forecasts and model.get("promising") else None, "wallet": _wallet_view(_wallet(data, model["model_id"]), current_price)})
    if models:
        _save(data)
    equity = sum(model["wallet"]["equity"] for model in model_views)
    initial = sum(model["wallet"]["initial_balance"] for model in model_views)
    cash = sum(model["wallet"]["cash"] for model in model_views)
    return {"initial_balance": initial, "cash": cash, "equity": equity, "realized_pnl": sum(m["wallet"]["realized_pnl"] for m in model_views), "unrealized_pnl": sum(m["wallet"]["unrealized_pnl"] for m in model_views), "return_pct": equity / initial - 1.0 if initial else 0.0, "current_price": current_price, "models": model_views, "controller": data.get("controller", {}), "virtual_only": True}


def _close_wallet(wallet: dict[str, Any], model_id: str, price: float, reason: str) -> float:
    position = wallet.get("position")
    if position is None:
        return 0.0
    move = price / position["entry_price"] - 1.0
    pnl = position["notional"] * (move if position["side"] == "long" else -move)
    wallet["cash"] += position["notional"] + pnl
    wallet["trades"].append({"model_id": model_id, **position, "exit_price": price, "pnl": pnl, "reason": reason, "closed_at": datetime.now(timezone.utc).isoformat()})
    wallet["position"] = None
    return pnl


def collect_inputs(market: MarketDataService, signal_service: SignalService, volatility: ModelRegistry) -> StrategyInputs:
    """Live inputs of the strategies; a missing piece is left None and noted (the rule then skips)."""
    candles = market.get_ohlcv("BTCUSDT", "1h")
    inputs = StrategyInputs(now=utc_now(), price=float(candles["close"].iloc[-1]), candles=candles[["timestamp", "close"]])
    try:
        response = signal_service.get_signals("BTCUSDT", threshold=0.50)
        for item in response.horizons:
            if item.horizon == "4h":
                inputs.direction_4h = {"p_up": item.p_up, "confidence": item.confidence, "has_skill": item.has_skill,
                                       "as_of": response.data_as_of.isoformat(), "model_id": item.model_version}
    except Exception as exc:  # noqa: BLE001 - one missing input must not stop the others
        inputs.notes.append(f"direction unavailable: {type(exc).__name__}")
    try:
        from app.api.routes.volatility import _forecast, _latest_by_variant

        meta = next((m for m in _latest_by_variant(volatility) if m.get("variant") == "move_1pct_4h:candles"), None)
        if meta is not None:
            forecast = _forecast(volatility, meta, candles, None)
            if forecast.get("available"):
                inputs.p_move_4h = forecast["p_move"]
    except Exception as exc:  # noqa: BLE001
        inputs.notes.append(f"volatility unavailable: {type(exc).__name__}")
    try:
        from app.api.routes.euphoria import _frame

        frame = _frame(_resolve(get_settings().data_dir))
        valid = None if frame is None else frame.dropna(subset=["euphoria"])
        if valid is not None and len(valid):
            row = valid.iloc[-1]
            inputs.euphoria = {"value": float(row["euphoria"]), "as_of": (row["timestamp"] + pd.Timedelta(hours=1)).isoformat()}
    except Exception as exc:  # noqa: BLE001
        inputs.notes.append(f"euphoria unavailable: {type(exc).__name__}")
    return inputs


def run_strategy_tick() -> list[str]:
    """One engine step with the app's shared services (called by the background loop)."""
    inputs = collect_inputs(get_market_data_service(), get_signal_service(), get_volatility_registry())
    with _lock:
        data = _load()
        controller = data.setdefault("controller", {"enabled": True, "last_tick": None, "last_action": "", "threshold": 0.60})
        if not controller.get("enabled"):
            return []
        actions = tick(data, inputs)
        controller["last_tick"] = inputs.now.isoformat()
        note = f" ({'; '.join(inputs.notes)})" if inputs.notes else ""
        controller["last_action"] = "; ".join(actions) if actions else "no qualifying signal" + note
        _save(data)
    return actions


def strategy_loop(stop: threading.Event, interval_seconds: int) -> None:
    while not stop.is_set():
        try:
            run_strategy_tick()
        except Exception:  # noqa: BLE001 - the loop must survive network or data errors
            logger.exception("Paper strategy tick failed")
        stop.wait(interval_seconds)


class TradeRequest(BaseModel):
    model_id: str = Field(min_length=1)
    side: Literal["long", "short"]
    notional: float = Field(gt=10, le=10000)


@router.get("/overview")
def overview(registry: Registry, barrier: BarrierRegistry, experiments: ExperimentRegistry, market: MarketData, signal_service: Signal) -> dict[str, Any]:
    """Read-only: trading happens in the background loop, never as a side effect of viewing."""
    models = _models(registry, barrier, experiments)
    price = _price(market)
    try:
        signal_response = signal_service.get_signals("BTCUSDT", threshold=0.50)
        forecasts = {item.model_version: {"horizon": item.horizon, "direction": item.direction, "p_up": item.p_up, "confidence": item.confidence, "is_signal": item.is_signal, "has_skill": item.has_skill, "target_time": item.target_time.isoformat(), "status": item.status} for item in signal_response.horizons}
    except Exception:
        forecasts = {}
    forecasts.update(_state_forecasts(experiments, market, models))
    forecasts.update(_barrier_forecasts(barrier, market, models))
    with _lock:  # load inside the lock: the background loop may have traded meanwhile
        data = _load()
        result = _overview(data, price, models, forecasts)
    wallets = data.get("strategies") or {}
    result["strategies"] = [strategy_view(s, wallets.get(s.id) or new_wallet(), price) for s in STRATEGIES]
    return result


@router.post("/trade")
def open_trade(request: TradeRequest, registry: Registry, barrier: BarrierRegistry, experiments: ExperimentRegistry, market: MarketData) -> dict[str, Any]:
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
def close_trade(model_id: str, registry: Registry, barrier: BarrierRegistry, experiments: ExperimentRegistry, market: MarketData) -> dict[str, Any]:
    with _lock:
        data = _load()
        wallet = _wallet(data, model_id)
        position = wallet.get("position")
        if position is None:
            raise HTTPException(status_code=404, detail="Позиция не найдена")
        price = _price(market)
        _close_wallet(wallet, model_id, price, "manual")
        _save(data)
        return _overview(data, price, _models(registry, barrier, experiments))


@router.post("/reset")
def reset_portfolio(
    x_operations_token: str | None = Header(default=None),
    x_operations_token_b64: str | None = Header(default=None),
) -> dict[str, str]:
    _check_token(x_operations_token, x_operations_token_b64)
    with _lock:
        _save(_empty())
    return {"status": "reset", "balance_per_model": f"{STARTING_BALANCE:.2f}"}


class ControllerRequest(BaseModel):
    enabled: bool


@router.post("/controller")
def set_controller(
    request: ControllerRequest,
    x_operations_token: str | None = Header(default=None),
    x_operations_token_b64: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_token(x_operations_token, x_operations_token_b64)
    with _lock:
        data = _load()
        controller = data.setdefault("controller", {"enabled": False, "last_tick": None, "last_action": "off", "threshold": 0.60})
        controller["enabled"] = request.enabled
        controller["last_action"] = "enabled; waiting for qualifying signal" if request.enabled else "off"
        _save(data)
        return controller
