"""Aggregate a stream of events into one row per minute.

A minute is emitted when the first event of a later minute arrives, so only
complete minutes are written. The first minute after a (re)connect was only
partially observed, so it is discarded as well: a minute is written only if
the collector saw it from its very beginning.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.data.live.messages import BookSnapshot, MarkPrice, Trade

MINUTE_MS = 60_000

Row = dict[str, Any]


def minute_of(time_ms: int) -> int:
    return time_ms - time_ms % MINUTE_MS


def minute_timestamp(minute_ms: int) -> pd.Timestamp:
    return pd.Timestamp(minute_ms, unit="ms", tz="UTC")


class MinuteAggregator:
    """Base class: collects events of the current minute and emits finished rows."""

    def __init__(self) -> None:
        self._minute: int | None = None
        self._partial = True  # current minute was not observed from its start
        self.late_events = 0
        self.partial_minutes_dropped = 0

    def _roll(self, time_ms: int) -> list[Row]:
        """Advance to the minute of `time_ms`; return the finished row if the minute changed."""
        minute = minute_of(time_ms)
        if self._minute is None:
            self._minute = minute
            self._partial = True
            self._start()
            return []
        if minute < self._minute:
            self.late_events += 1
            raise _LateEvent
        if minute == self._minute:
            return []
        finished = None if self._partial else self._finish(self._minute)
        if self._partial:
            self.partial_minutes_dropped += 1
        self._minute = minute
        self._partial = False
        self._start()
        return [finished] if finished is not None else []

    def reset(self) -> None:
        self._minute = None
        self._partial = True

    def _start(self) -> None:
        raise NotImplementedError

    def _finish(self, minute_ms: int) -> Row | None:
        raise NotImplementedError


class _LateEvent(Exception):
    pass


@dataclass
class _TradeState:
    trades: int = 0
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    buy_quote: float = 0.0
    sell_quote: float = 0.0
    large_buy_quote: float = 0.0
    large_sell_quote: float = 0.0
    open: float | None = None
    high: float = float("-inf")
    low: float = float("inf")
    close: float | None = None


class TradeFlowAggregator(MinuteAggregator):
    """Aggressive buy vs sell volume per minute (order flow).

    In Binance aggTrade, m=True means the buyer was the maker, i.e. the
    aggressive side (taker) was the seller.
    """

    def __init__(self, large_trade_quote: float = 100_000.0) -> None:
        super().__init__()
        self.large_trade_quote = large_trade_quote
        self._state = _TradeState()

    def _start(self) -> None:
        self._state = _TradeState()

    def add(self, trade: Trade) -> list[Row]:
        try:
            rows = self._roll(trade.time_ms)
        except _LateEvent:
            return []
        s = self._state
        s.trades += 1
        quote = trade.quote
        large = quote >= self.large_trade_quote
        if trade.buyer_is_maker:
            s.sell_qty += trade.qty
            s.sell_quote += quote
            if large:
                s.large_sell_quote += quote
        else:
            s.buy_qty += trade.qty
            s.buy_quote += quote
            if large:
                s.large_buy_quote += quote
        if s.open is None:
            s.open = trade.price
        s.high = max(s.high, trade.price)
        s.low = min(s.low, trade.price)
        s.close = trade.price
        return rows

    def _finish(self, minute_ms: int) -> Row | None:
        s = self._state
        if s.trades == 0:
            return None
        volume = s.buy_qty + s.sell_qty
        quote = s.buy_quote + s.sell_quote
        return {
            "timestamp": minute_timestamp(minute_ms),
            "trades": s.trades,
            "open": s.open,
            "high": s.high,
            "low": s.low,
            "close": s.close,
            "vwap": quote / volume if volume else None,
            "buy_qty": s.buy_qty,
            "sell_qty": s.sell_qty,
            "buy_quote": s.buy_quote,
            "sell_quote": s.sell_quote,
            "large_buy_quote": s.large_buy_quote,
            "large_sell_quote": s.large_sell_quote,
            "flow_imbalance": (s.buy_quote - s.sell_quote) / quote if quote else None,
        }


def book_metrics(book: BookSnapshot, top: int = 5) -> dict[str, float] | None:
    """Spread and bid/ask quantity imbalance of one order book snapshot."""
    if not book.bids or not book.asks:
        return None
    best_bid, best_ask = book.bids[0][0], book.asks[0][0]
    mid = (best_bid + best_ask) / 2
    bid_top = sum(q for _, q in book.bids[:top])
    ask_top = sum(q for _, q in book.asks[:top])
    bid_all = sum(q for _, q in book.bids)
    ask_all = sum(q for _, q in book.asks)
    return {
        "mid": mid,
        "spread_bps": (best_ask - best_bid) / mid * 10_000,
        "imbalance_top5": (bid_top - ask_top) / (bid_top + ask_top) if bid_top + ask_top else 0.0,
        "imbalance_all": (bid_all - ask_all) / (bid_all + ask_all) if bid_all + ask_all else 0.0,
        "bid_qty": bid_all,
        "ask_qty": ask_all,
    }


@dataclass
class _BookState:
    snapshots: int = 0
    sums: dict[str, float] = field(default_factory=dict)
    last_mid: float | None = None


class OrderBookAggregator(MinuteAggregator):
    """Average order book metrics over the snapshots received in a minute."""

    def __init__(self) -> None:
        super().__init__()
        self._state = _BookState()

    def _start(self) -> None:
        self._state = _BookState()

    def add(self, book: BookSnapshot, time_ms: int) -> list[Row]:
        metrics = book_metrics(book)
        try:
            rows = self._roll(time_ms)
        except _LateEvent:
            return []
        if metrics is None:
            return rows
        s = self._state
        s.snapshots += 1
        for key, value in metrics.items():
            s.sums[key] = s.sums.get(key, 0.0) + value
        s.last_mid = metrics["mid"]
        return rows

    def _finish(self, minute_ms: int) -> Row | None:
        s = self._state
        if s.snapshots == 0:
            return None
        means = {key: value / s.snapshots for key, value in s.sums.items() if key != "mid"}
        return {
            "timestamp": minute_timestamp(minute_ms),
            "snapshots": s.snapshots,
            "mid_close": s.last_mid,
            **means,
        }


class MarkPriceAggregator(MinuteAggregator):
    """Last mark/index price and funding rate of each minute, plus mean basis."""

    def __init__(self) -> None:
        super().__init__()
        self._last: MarkPrice | None = None
        self._basis_sum = 0.0
        self._count = 0

    def _start(self) -> None:
        self._last = None
        self._basis_sum = 0.0
        self._count = 0

    def add(self, mark: MarkPrice) -> list[Row]:
        try:
            rows = self._roll(mark.time_ms)
        except _LateEvent:
            return []
        self._last = mark
        if mark.index_price:
            self._basis_sum += (mark.mark_price - mark.index_price) / mark.index_price
            self._count += 1
        return rows

    def _finish(self, minute_ms: int) -> Row | None:
        if self._last is None:
            return None
        m = self._last
        return {
            "timestamp": minute_timestamp(minute_ms),
            "mark_price": m.mark_price,
            "index_price": m.index_price,
            "funding_rate": m.funding_rate,
            "next_funding_time": pd.Timestamp(m.next_funding_time_ms, unit="ms", tz="UTC")
            if m.next_funding_time_ms
            else None,
            "basis_mean": self._basis_sum / self._count if self._count else None,
        }
