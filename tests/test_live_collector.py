import asyncio
import json
from contextlib import asynccontextmanager

import pandas as pd
import pytest

from app.data.live.aggregators import (
    MarkPriceAggregator,
    OrderBookAggregator,
    TradeFlowAggregator,
    book_metrics,
)
from app.data.live.collector import LiveCollector
from app.data.live.messages import (
    BookSnapshot,
    Trade,
    depth_event_time,
    parse_agg_trade,
    parse_depth,
    parse_liquidation,
    parse_mark_price,
)
from app.data.live.writer import LiveDataWriter, load_live

T0 = 1_794_336_000_000  # a minute boundary
MIN = 60_000

# Real payload shapes captured from Binance
SPOT_TRADE = {"e": "aggTrade", "E": T0, "s": "BTCUSDT", "a": 1, "p": "73482.00", "q": "0.5", "f": 1, "l": 1, "T": T0, "m": True, "M": True}
SPOT_DEPTH = {"lastUpdateId": 1, "bids": [["100.0", "3"], ["99.9", "1"]], "asks": [["100.1", "1"], ["100.2", "1"]]}
FUT_DEPTH = {"e": "depthUpdate", "E": T0 + 5, "T": T0 + 4, "s": "BTCUSDT", "b": [["100.0", "1"]], "a": [["100.2", "3"]]}
MARK = {"e": "markPriceUpdate", "E": T0, "s": "BTCUSDT", "p": "73435.2", "ap": "73435.2", "P": "73413.6", "i": "73468.6", "r": "0.00006437", "T": T0 + 3_600_000}
LIQ = {"e": "forceOrder", "E": T0, "o": {"s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "0.010", "p": "73208.8", "ap": "73491.2", "X": "FILLED", "l": "0.010", "z": "0.010", "T": T0}}


def trade(t: int, price: float, qty: float, seller_aggressor: bool) -> Trade:
    return Trade(time_ms=t, price=price, qty=qty, buyer_is_maker=seller_aggressor)


# --- parsers -----------------------------------------------------------------------


def test_parsers() -> None:
    t = parse_agg_trade(SPOT_TRADE)
    assert (t.time_ms, t.price, t.qty, t.buyer_is_maker) == (T0, 73482.0, 0.5, True)

    spot = parse_depth(SPOT_DEPTH)
    fut = parse_depth(FUT_DEPTH)
    assert spot.bids[0] == (100.0, 3.0) and fut.asks[0] == (100.2, 3.0)
    assert depth_event_time(SPOT_DEPTH) is None and depth_event_time(FUT_DEPTH) == T0 + 5

    m = parse_mark_price(MARK)
    assert m.funding_rate == pytest.approx(0.00006437) and m.index_price == pytest.approx(73468.6)

    liq = parse_liquidation(LIQ)
    assert liq.side == "SELL" and liq.quote == pytest.approx(73491.2 * 0.01)


# --- aggregators ------------------------------------------------------------------


def test_trade_flow_minute_row() -> None:
    agg = TradeFlowAggregator(large_trade_quote=1000)
    agg.add(trade(T0 - 1, 1.0, 1.0, False))  # partial minute before T0: must be dropped
    assert agg.add(trade(T0 + 1_000, 100.0, 2.0, seller_aggressor=False)) == []  # first full minute starts
    agg.add(trade(T0 + 2_000, 102.0, 20.0, seller_aggressor=True))  # 2040 quote, large sell
    agg.add(trade(T0 + 3_000, 99.0, 1.0, seller_aggressor=False))
    rows = agg.add(trade(T0 + MIN + 1, 101.0, 1.0, False))

    assert len(rows) == 1
    row = rows[0]
    assert row["timestamp"] == pd.Timestamp(T0, unit="ms", tz="UTC")
    assert row["trades"] == 3
    assert (row["open"], row["high"], row["low"], row["close"]) == (100.0, 102.0, 99.0, 99.0)
    assert row["buy_qty"] == pytest.approx(3.0) and row["sell_qty"] == pytest.approx(20.0)
    assert row["buy_quote"] == pytest.approx(299.0) and row["sell_quote"] == pytest.approx(2040.0)
    assert row["large_sell_quote"] == pytest.approx(2040.0) and row["large_buy_quote"] == 0
    assert row["flow_imbalance"] == pytest.approx((299 - 2040) / 2339)
    assert row["vwap"] == pytest.approx(2339 / 23)
    assert agg.partial_minutes_dropped == 1


def test_first_minute_after_start_and_reset_is_dropped() -> None:
    agg = TradeFlowAggregator()
    agg.add(trade(T0 + 30_000, 100, 1, False))  # started mid-minute
    assert agg.add(trade(T0 + MIN, 100, 1, False)) == []  # partial minute dropped
    rows = agg.add(trade(T0 + 2 * MIN, 100, 1, False))
    assert [r["timestamp"] for r in rows] == [pd.Timestamp(T0 + MIN, unit="ms", tz="UTC")]

    agg.reset()  # reconnect in the middle of a minute
    agg.add(trade(T0 + 2 * MIN + 40_000, 100, 1, False))
    assert agg.add(trade(T0 + 3 * MIN, 100, 1, False)) == []


def test_late_events_are_ignored() -> None:
    agg = TradeFlowAggregator()
    agg.add(trade(T0 + MIN, 100, 1, False))
    assert agg.add(trade(T0 + 10, 100, 1, False)) == []
    assert agg.late_events == 1


def test_book_metrics_and_aggregation() -> None:
    book = BookSnapshot(bids=[(100.0, 3.0), (99.0, 1.0)], asks=[(101.0, 1.0), (102.0, 1.0)])
    m = book_metrics(book)
    assert m["mid"] == 100.5
    assert m["spread_bps"] == pytest.approx(1 / 100.5 * 10_000)
    assert m["imbalance_all"] == pytest.approx((4 - 2) / 6)
    assert book_metrics(BookSnapshot([], [])) is None

    agg = OrderBookAggregator()
    agg.add(book, T0 - 1)
    agg.add(book, T0)
    agg.add(BookSnapshot(bids=[(100.0, 1.0)], asks=[(101.0, 3.0)]), T0 + 500)
    rows = agg.add(book, T0 + MIN)
    assert rows[0]["snapshots"] == 2
    assert rows[0]["imbalance_top5"] == pytest.approx(((4 - 2) / 6 + (1 - 3) / 4) / 2)


def test_mark_price_aggregation() -> None:
    agg = MarkPriceAggregator()
    agg.add(parse_mark_price({**MARK, "E": T0 - 1}))
    agg.add(parse_mark_price(MARK))
    rows = agg.add(parse_mark_price({**MARK, "E": T0 + MIN}))
    assert rows[0]["funding_rate"] == pytest.approx(0.00006437)
    assert rows[0]["basis_mean"] == pytest.approx((73435.2 - 73468.6) / 73468.6)
    assert rows[0]["next_funding_time"] == pd.Timestamp(T0 + 3_600_000, unit="ms", tz="UTC")


# --- writer ------------------------------------------------------------------------


def minute_rows(start: int, count: int, value: float = 1.0) -> list[dict]:
    return [{"timestamp": pd.Timestamp(start + i * MIN, unit="ms", tz="UTC"), "x": value} for i in range(count)]


def test_writer_splits_by_hour_and_merges_on_restart(tmp_path) -> None:
    writer = LiveDataWriter(tmp_path, "btcusdt")
    hour_start = T0 - T0 % 3_600_000
    writer.add("trades_spot", minute_rows(hour_start + 50 * MIN, 20))  # crosses into the next hour
    assert writer.flush() == 20

    files = sorted((tmp_path / "trades_spot" / "BTCUSDT").glob("*/*.parquet"))
    assert len(files) == 2

    # A restarted collector rewrites an overlapping minute with a newer value
    restarted = LiveDataWriter(tmp_path, "BTCUSDT")
    restarted.add("trades_spot", minute_rows(hour_start + 69 * MIN, 3, value=2.0))
    restarted.flush()

    df = load_live(tmp_path, "trades_spot", "BTCUSDT")
    assert len(df) == 22
    assert df["timestamp"].is_unique and df["timestamp"].is_monotonic_increasing
    assert df["x"].iloc[-3:].tolist() == [2.0, 2.0, 2.0]


def test_writer_keeps_simultaneous_events(tmp_path) -> None:
    writer = LiveDataWriter(tmp_path, "BTCUSDT")
    ts = pd.Timestamp(T0, unit="ms", tz="UTC")
    writer.add(
        "connection_events",
        [{"timestamp": ts, "connection": c, "event": "connected", "detail": ""} for c in ("spot", "futures_market")],
    )
    writer.flush()
    assert len(load_live(tmp_path, "connection_events", "BTCUSDT")) == 2


# --- collector ---------------------------------------------------------------------


def wrap(stream: str, data: dict) -> str:
    return json.dumps({"stream": stream, "data": data})


def test_collector_routes_messages(tmp_path) -> None:
    clock = {"ms": T0}
    collector = LiveCollector(LiveDataWriter(tmp_path, "BTCUSDT"), clock_ms=lambda: clock["ms"])
    spot, futures_market, futures_public = collector.connections

    for minute in range(3):
        t = T0 + minute * MIN + 1
        clock["ms"] = t
        collector.handle_message(spot, wrap("btcusdt@aggTrade", {**SPOT_TRADE, "T": t}))
        collector.handle_message(spot, wrap("btcusdt@depth20@1000ms", SPOT_DEPTH))
        collector.handle_message(futures_market, wrap("btcusdt@markPrice@1s", {**MARK, "E": t}))
        collector.handle_message(futures_public, wrap("btcusdt@depth20@500ms", {**FUT_DEPTH, "E": t}))

    # liquidation for another symbol is ignored
    collector.handle_message(futures_market, wrap("!forceOrder@arr", {**LIQ, "o": {**LIQ["o"], "s": "ETHUSDT"}}))
    collector.handle_message(futures_market, wrap("!forceOrder@arr", LIQ))
    collector.handle_message(spot, "not json")
    collector.handle_message(spot, wrap("btcusdt@aggTrade", {"broken": True}))
    collector.flush()

    for dataset in ("trades_spot", "book_spot", "mark_price", "book_futures"):
        df = load_live(tmp_path, dataset, "BTCUSDT")
        # minute 0 is partial (dropped), minute 2 is still open -> only minute 1 written
        assert df["timestamp"].tolist() == [pd.Timestamp(T0 + MIN, unit="ms", tz="UTC")], dataset
    assert len(load_live(tmp_path, "liquidations", "BTCUSDT")) == 1
    assert collector.errors == 2

    status = json.loads((tmp_path / "_status_BTCUSDT.json").read_text(encoding="utf-8"))
    assert status["connections"]["spot"]["messages"] == 8
    assert status["rows_written"]["trades_spot"] == 1


def test_collector_reconnects_and_logs_events(tmp_path) -> None:
    ticks = iter(range(T0, T0 + 10_000_000, 7))  # strictly increasing fake clock
    collector = LiveCollector(LiveDataWriter(tmp_path, "BTCUSDT"), clock_ms=lambda: next(ticks))
    connection = collector.connections[0]
    attempts = {"n": 0}

    @asynccontextmanager
    async def fake_connect(url):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("network down")

        async def messages():
            yield wrap("btcusdt@aggTrade", SPOT_TRADE)
            if attempts["n"] >= 3:
                await asyncio.Event().wait()  # stay connected until cancelled

        yield messages()

    async def scenario():
        task = asyncio.create_task(collector.run_connection(connection, connect=fake_connect, max_backoff=0.01))
        while attempts["n"] < 3:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    import app.data.live.collector as module

    original_sleep = module.asyncio.sleep

    async def fast_sleep(seconds):
        await original_sleep(min(seconds, 0.001))

    module.asyncio.sleep = fast_sleep
    try:
        asyncio.run(scenario())
    finally:
        module.asyncio.sleep = original_sleep

    collector.writer.flush()
    events = load_live(tmp_path, "connection_events", "BTCUSDT")["event"].tolist()
    assert events.count("connected") == 2
    assert events.count("disconnected") == 2  # network error + server closed the stream
    assert events[-1] == "stopped"
