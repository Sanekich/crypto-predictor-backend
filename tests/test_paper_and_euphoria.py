"""Paper-trading strategy rules (forward test) and the euphoria / panic index."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.ml.euphoria.index import COMPONENTS, build_euphoria_frame
from app.ml.euphoria.study import event_positions, run_study
from app.services.paper_strategies import FEE_RATE, POSITION_FRACTION, STRATEGIES, StrategyInputs, new_wallet, strategy_view, tick
from tests.factories import make_random_ohlcv

HOUR = timedelta(hours=1)
T0 = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)  # close time of the signal candle


def candles(prices: dict[datetime, float]) -> pd.DataFrame:
    """prices keyed by candle CLOSE time."""
    return pd.DataFrame({"timestamp": [pd.Timestamp(t - HOUR) for t in prices], "close": list(prices.values())})


def direction(p_up: float, as_of: datetime = T0, skill: bool = True) -> dict:
    return {"p_up": p_up, "confidence": max(p_up, 1 - p_up), "has_skill": skill, "as_of": as_of.isoformat(), "model_id": "m4h"}


DIRECTION_ONLY = tuple(s for s in STRATEGIES if s.id == "direction-4h-60")
WITH_VOL = tuple(s for s in STRATEGIES if s.id == "direction-4h-60-vol")
PANIC = tuple(s for s in STRATEGIES if s.id == "panic-bounce-24h")


# --- strategy rules -------------------------------------------------------------------


def test_opens_on_fresh_confident_signal_and_closes_exactly_at_horizon_with_fees() -> None:
    state: dict = {}
    prices = {T0: 100.0, T0 + 4 * HOUR: 110.0, T0 + 5 * HOUR: 120.0}
    tick(state, StrategyInputs(now=T0 + timedelta(minutes=5), price=100.0, candles=candles({T0: 100.0}), direction_4h=direction(0.65)), DIRECTION_ONLY)
    wallet = state["strategies"]["direction-4h-60"]
    position = wallet["position"]
    assert position["side"] == "long" and position["entry_price"] == 100.0
    assert position["expires_at"] == (T0 + timedelta(minutes=5) + 4 * HOUR).isoformat() or position["expires_at"] == (T0 + 4 * HOUR).isoformat()

    # one hour after the horizon the exit uses the candle that closed AT the horizon (110), not the latest price (120)
    tick(state, StrategyInputs(now=T0 + 5 * HOUR, price=120.0, candles=candles(prices)), DIRECTION_ONLY)
    trade = wallet["trades"][0]
    assert wallet["position"] is None and trade["exit_price"] == 110.0 and trade["reason"] == "horizon"
    notional = 10_000 * POSITION_FRACTION
    expected = notional * 0.10 - notional * FEE_RATE - notional * 1.10 * FEE_RATE
    assert trade["pnl"] == pytest.approx(expected)
    assert wallet["cash"] == pytest.approx(10_000 + expected)
    assert trade["correct"] is True


def test_waits_for_exit_candle_then_falls_back_when_data_is_stale() -> None:
    state: dict = {}
    tick(state, StrategyInputs(now=T0, price=100.0, candles=candles({T0: 100.0}), direction_4h=direction(0.30)), DIRECTION_ONLY)
    assert state["strategies"]["direction-4h-60"]["position"]["side"] == "short"
    tick(state, StrategyInputs(now=T0 + 5 * HOUR, price=95.0, candles=candles({T0: 100.0})), DIRECTION_ONLY)
    assert state["strategies"]["direction-4h-60"]["position"] is not None  # exit candle not known yet
    tick(state, StrategyInputs(now=T0 + 8 * HOUR, price=95.0, candles=candles({T0: 100.0})), DIRECTION_ONLY)
    trade = state["strategies"]["direction-4h-60"]["trades"][0]
    assert trade["exit_price"] == 95.0 and trade["correct"] is True


@pytest.mark.parametrize("signal", [
    direction(0.58),                      # below 60% confidence
    direction(0.70, skill=False),         # model without walk-forward skill
    direction(0.70, as_of=T0 - 2 * HOUR), # stale signal: the candle closed 2 hours ago
])
def test_no_trade_without_a_qualifying_signal(signal) -> None:
    state: dict = {}
    tick(state, StrategyInputs(now=T0 + timedelta(minutes=10), price=100.0, candles=candles({T0: 100.0, T0 - 2 * HOUR: 99.0}), direction_4h=signal), DIRECTION_ONLY)
    assert state["strategies"]["direction-4h-60"]["position"] is None


def test_volatility_filter_and_no_double_entry_on_the_same_signal() -> None:
    state: dict = {}
    calm = StrategyInputs(now=T0, price=100.0, candles=candles({T0: 100.0}), direction_4h=direction(0.7), p_move_4h=0.4)
    tick(state, calm, WITH_VOL)
    assert state["strategies"]["direction-4h-60-vol"]["position"] is None
    tick(state, StrategyInputs(now=T0, price=100.0, candles=candles({T0: 100.0}), direction_4h=direction(0.7), p_move_4h=0.8), WITH_VOL)
    wallet = state["strategies"]["direction-4h-60-vol"]
    assert wallet["position"] is not None
    # closed at the horizon, the same (old) signal must not be traded again
    prices = {T0: 100.0, T0 + 4 * HOUR: 101.0}
    tick(state, StrategyInputs(now=T0 + 4 * HOUR + timedelta(minutes=1), price=101.0, candles=candles(prices), direction_4h=direction(0.7), p_move_4h=0.8), WITH_VOL)
    assert wallet["position"] is None and len(wallet["trades"]) == 1


def test_panic_strategy_buys_only_when_the_zone_is_entered() -> None:
    state: dict = {}
    base = dict(price=100.0, candles=candles({T0: 100.0}))
    tick(state, StrategyInputs(now=T0, euphoria={"value": -1.0, "as_of": "d1"}, **base), PANIC)
    assert state["strategies"]["panic-bounce-24h"]["position"] is None
    tick(state, StrategyInputs(now=T0 + HOUR, euphoria={"value": -1.8, "as_of": "d2"}, **base), PANIC)
    position = state["strategies"]["panic-bounce-24h"]["position"]
    assert position["side"] == "long" and position["expires_at"] == (T0 + 25 * HOUR).isoformat()


def test_strategy_view_reports_win_rate_and_fees() -> None:
    wallet = new_wallet()
    wallet["trades"] = [{"correct": True, "return_pct": 0.01, "fee_entry": 1, "fee_exit": 1}, {"correct": False, "return_pct": -0.02, "fee_entry": 1, "fee_exit": 1}]
    view = strategy_view(STRATEGIES[0], wallet, 100.0)
    assert view["trades_count"] == 2 and view["win_rate"] == 0.5 and view["fees_paid"] == 4
    assert view["avg_trade_return"] == pytest.approx(-0.005)


# --- euphoria index ---------------------------------------------------------------------


def derivatives_for(ohlcv: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start, end = ohlcv["timestamp"].iloc[0], ohlcv["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    eight = pd.date_range(start, end, freq="8h", tz="UTC")
    five = pd.date_range(start, end, freq="5min", tz="UTC")
    days = pd.date_range(start.floor("D"), end, freq="D", tz="UTC")
    return {
        "funding": pd.DataFrame({"available_at": eight, "funding_rate": rng.normal(1e-4, 5e-5, len(eight))}),
        "metrics": pd.DataFrame({
            "available_at": five, "open_interest": 1.0,
            "open_interest_value": 1e9 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(five)))),
            "top_trader_account_ratio": 1.0, "top_trader_position_ratio": 1.0,
            "global_account_ratio": rng.uniform(0.8, 2.0, len(five)), "taker_buy_sell_ratio": 1.0,
        }),
        "fear_greed": pd.DataFrame({"timestamp": days, "available_at": days + pd.Timedelta(hours=24), "fear_greed": rng.uniform(10, 90, len(days))}),
    }


def test_euphoria_index_has_no_lookahead() -> None:
    ohlcv = make_random_ohlcv(24 * 80, seed=3)  # 30d average + 30d z-score warm-up
    data = derivatives_for(ohlcv)
    base = build_euphoria_frame(ohlcv, data)
    assert base["euphoria"].notna().sum() > 24 * 10
    assert set(f"z_{c}" for c in COMPONENTS) <= set(base.columns)

    cut = ohlcv["timestamp"].iloc[1600] + pd.Timedelta(hours=1)
    changed = {}
    for name, frame in data.items():
        frame = frame.copy()
        later = frame["available_at"] > cut
        numeric = [c for c in frame.columns if c not in ("timestamp", "available_at")]
        frame.loc[later, numeric] = frame.loc[later, numeric] * 3 + 1
        changed[name] = frame
    shifted = ohlcv.copy()
    shifted.loc[1601:, ["open", "high", "low", "close"]] *= 2
    after = build_euphoria_frame(shifted, changed)
    pd.testing.assert_series_equal(base["euphoria"].iloc[:1601], after["euphoria"].iloc[:1601])


def test_euphoria_requires_its_datasets() -> None:
    ohlcv = make_random_ohlcv(100)
    with pytest.raises(ValueError, match="fear_greed"):
        build_euphoria_frame(ohlcv, {k: v for k, v in derivatives_for(ohlcv).items() if k != "fear_greed"})


def test_events_are_zone_entries_spaced_by_horizon() -> None:
    zone = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=bool)
    assert event_positions(zone, 3).tolist() == [1, 4, 8]
    assert event_positions(zone, 4).tolist() == [1, 8]


def test_study_measures_reversal_hit_rate() -> None:
    n = 2000
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.full(n, 100.0)
    score = np.zeros(n)
    for start in range(100, 1800, 200):  # panic spikes followed by a rise
        score[start:start + 3] = -2.5
        close[start + 1:] *= 1.02
    frame = pd.DataFrame({"timestamp": ts, "close": close, "euphoria": score})
    row = next(r for r in run_study(frame)["results"] if r["zone"] == "panic" and r["threshold"] == 2.0 and r["horizon"] == "24h")
    assert row["events"] == 9 and row["hit_rate"] == 1.0 and not row["passes_gate"]
