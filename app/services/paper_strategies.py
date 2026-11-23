"""Forward test: virtual strategies that trade the signals we found on history.

History can be overfitted; the future cannot. Each strategy has its own virtual
wallet and follows fixed rules, so after weeks of running its win rate is honest
out-of-sample evidence:

- a position is opened only on a fresh signal (the signal candle closed < 1 hour ago),
  at that candle's close;
- it is closed exactly at the end of its horizon, at the close of the candle that
  ends there (never "when it looks good");
- a fee is paid on entry and on exit (Binance futures taker, 0.05% per side);
- one position at a time per strategy, 25% of the wallet's cash.

The rules are pure functions of `StrategyInputs`, so they are unit-tested without
the network; `collect_inputs` (in the route module) gathers the live inputs.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.ml.evaluation.metrics import wilson_lower_bound

FEE_RATE = 0.0005
POSITION_FRACTION = 0.25
STARTING_BALANCE = 10_000.0
FRESH_SIGNAL = timedelta(hours=1)
STALE_EXIT = timedelta(hours=3)  # no candle for the exit time this long after expiry -> close at the last price


@dataclass(frozen=True)
class Strategy:
    id: str
    label: str
    description: str
    kind: str  # "direction" | "panic"
    horizon_hours: int
    min_confidence: float = 0.0
    min_p_move: float | None = None
    panic_threshold: float | None = None


STRATEGIES: tuple[Strategy, ...] = (
    Strategy("direction-4h-60", "Направление 4ч ≥ 60%",
             "Модель направления на 4 часа, уверенность от 60%. На истории: 764 сигнала, 60.5% попаданий.",
             "direction", 4, min_confidence=0.60),
    Strategy("direction-4h-60-vol", "Направление 4ч ≥ 60% + волатильность",
             "То же, но только когда модель волатильности ждёт движения ±1% за 4 часа (P ≥ 60%). Цель — более крупные движения.",
             "direction", 4, min_confidence=0.60, min_p_move=0.60),
    Strategy("direction-4h-55-vol", "Направление 4ч ≥ 55% + волатильность",
             "Больше сигналов: уверенность от 55% при ожидаемом сильном движении.",
             "direction", 4, min_confidence=0.55, min_p_move=0.60),
    Strategy("panic-bounce-24h", "Отскок после паники",
             "Покупка на входе индекса эйфории в зону паники (≤ −1.5), выход через 24 часа. На истории: 25 событий, 76%.",
             "panic", 24, panic_threshold=-1.5),
)


@dataclass
class StrategyInputs:
    now: datetime
    price: float  # close of the last closed candle
    candles: pd.DataFrame  # timestamp (open time), close
    direction_4h: dict[str, Any] | None = None  # p_up, confidence, has_skill, as_of, model_id
    p_move_4h: float | None = None  # P(+-1% within 4h) for the same candle
    euphoria: dict[str, Any] | None = None  # value, as_of
    notes: list[str] = field(default_factory=list)


def new_wallet() -> dict[str, Any]:
    return {"initial_balance": STARTING_BALANCE, "cash": STARTING_BALANCE, "position": None, "trades": [], "last_signal": None}


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _exit_price(inputs: StrategyInputs, expires_at: datetime) -> float | None:
    """Close of the candle that ends at expires_at; None if it is not known yet."""
    target_open = pd.Timestamp(expires_at) - pd.Timedelta(hours=1)
    match = inputs.candles.loc[inputs.candles["timestamp"] == target_open, "close"]
    if len(match):
        return float(match.iloc[-1])
    if inputs.now - expires_at >= STALE_EXIT:
        return inputs.price
    return None


def _close(wallet: dict[str, Any], price: float, closed_at: datetime, reason: str) -> dict[str, Any]:
    position = wallet["position"]
    move = price / position["entry_price"] - 1
    gross = position["notional"] * (move if position["side"] == "long" else -move)
    exit_value = position["notional"] + gross
    fee = exit_value * FEE_RATE
    wallet["cash"] += exit_value - fee
    trade = {
        **position,
        "exit_price": price,
        "closed_at": closed_at.isoformat(),
        "reason": reason,
        "fee_exit": fee,
        "pnl": gross - position["fee_entry"] - fee,  # net of both fees
        "return_pct": (gross - position["fee_entry"] - fee) / position["notional"],
        "correct": bool(move > 0) if position["side"] == "long" else bool(move < 0),
    }
    wallet["trades"].append(trade)
    wallet["position"] = None
    return trade


def _open(wallet: dict[str, Any], side: str, price: float, opened_at: datetime, horizon_hours: int, signal: dict[str, Any]) -> None:
    notional = wallet["cash"] * POSITION_FRACTION
    fee = notional * FEE_RATE
    wallet["cash"] -= notional + fee
    wallet["position"] = {
        "side": side,
        "notional": notional,
        "entry_price": price,
        "opened_at": opened_at.isoformat(),
        "expires_at": (opened_at + timedelta(hours=horizon_hours)).isoformat(),
        "fee_entry": fee,
        "signal": signal,
    }


def _decide(strategy: Strategy, inputs: StrategyInputs, state: dict[str, Any]) -> tuple[str, dict[str, Any], datetime] | None:
    """(side, signal description, entry time) if the strategy wants to open now."""
    if strategy.kind == "direction":
        d = inputs.direction_4h
        if not d or not d.get("has_skill") or d["confidence"] < strategy.min_confidence:
            return None
        as_of = _parse(d["as_of"])
        if inputs.now - as_of >= FRESH_SIGNAL:
            return None
        if strategy.min_p_move is not None and (inputs.p_move_4h is None or inputs.p_move_4h < strategy.min_p_move):
            return None
        side = "long" if d["p_up"] >= 0.5 else "short"
        return side, {"as_of": d["as_of"], "p_up": d["p_up"], "confidence": d["confidence"], "p_move": inputs.p_move_4h, "model_id": d.get("model_id")}, as_of
    if strategy.kind == "panic":
        e = inputs.euphoria
        if not e:
            return None
        previous = state.get("euphoria_previous") or {}
        is_new = previous.get("as_of") != e["as_of"]
        entered = e["value"] <= strategy.panic_threshold and (previous.get("value") is None or previous["value"] > strategy.panic_threshold)
        if is_new and entered:
            # the index lags (open interest is published daily): we act when we learn it, at today's price
            return "long", {"as_of": e["as_of"], "euphoria": e["value"]}, inputs.now
    return None


def tick(state: dict[str, Any], inputs: StrategyInputs, strategies: tuple[Strategy, ...] = STRATEGIES) -> list[str]:
    """Close expired positions, open new ones on fresh signals. Mutates and returns actions."""
    wallets = state.setdefault("strategies", {})
    actions: list[str] = []
    for strategy in strategies:
        wallet = wallets.setdefault(strategy.id, new_wallet())
        position = wallet.get("position")
        if position:
            expires_at = _parse(position["expires_at"])
            if inputs.now >= expires_at:
                price = _exit_price(inputs, expires_at)
                if price is not None:
                    trade = _close(wallet, price, expires_at, "horizon")
                    actions.append(f"{strategy.id}: close {trade['side']} {trade['return_pct']:+.2%}")
            if wallet.get("position"):
                continue
        decision = _decide(strategy, inputs, state)
        if decision is None:
            continue
        side, signal, entry_time = decision
        if wallet.get("last_signal") == signal["as_of"]:
            continue  # this signal was already traded
        price = inputs.price if strategy.kind == "panic" else _signal_price(inputs, signal["as_of"])
        if price is None:
            continue
        _open(wallet, side, price, entry_time, strategy.horizon_hours, signal)
        wallet["last_signal"] = signal["as_of"]
        actions.append(f"{strategy.id}: open {side} @ {price:.0f}")
    if inputs.euphoria:
        state["euphoria_previous"] = {"as_of": inputs.euphoria["as_of"], "value": inputs.euphoria["value"]}
    return actions


def _signal_price(inputs: StrategyInputs, as_of: str) -> float | None:
    target_open = pd.Timestamp(_parse(as_of)) - pd.Timedelta(hours=1)
    match = inputs.candles.loc[inputs.candles["timestamp"] == target_open, "close"]
    return float(match.iloc[-1]) if len(match) else None


def strategy_view(strategy: Strategy, wallet: dict[str, Any], price: float) -> dict[str, Any]:
    trades = wallet.get("trades", [])
    position = wallet.get("position")
    unrealized = 0.0
    position_view = None
    if position:
        move = price / position["entry_price"] - 1
        unrealized = position["notional"] * (move if position["side"] == "long" else -move)
        position_view = {**position, "current_price": price, "unrealized_pnl": unrealized}
    equity = wallet["cash"] + (position["notional"] + unrealized if position else 0.0)
    wins = sum(1 for t in trades if t.get("correct"))
    returns = [t["return_pct"] for t in trades]
    return {
        **asdict(strategy),
        "initial_balance": wallet["initial_balance"],
        "cash": wallet["cash"],
        "equity": equity,
        "return_pct": equity / wallet["initial_balance"] - 1,
        "trades_count": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) if trades else None,
        "win_rate_lower_95": wilson_lower_bound(wins, len(trades)),
        "avg_trade_return": sum(returns) / len(returns) if returns else None,
        "fees_paid": sum(t.get("fee_entry", 0) + t.get("fee_exit", 0) for t in trades) + (position["fee_entry"] if position else 0.0),
        "position": position_view,
        "recent_trades": trades[-10:][::-1],
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
