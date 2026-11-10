"""Parsers for Binance WebSocket payloads into small typed records."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Trade:
    time_ms: int
    price: float
    qty: float
    buyer_is_maker: bool  # True -> the aggressor was a seller

    @property
    def quote(self) -> float:
        return self.price * self.qty


@dataclass(frozen=True)
class BookSnapshot:
    bids: list[tuple[float, float]]  # best first: (price, qty)
    asks: list[tuple[float, float]]


@dataclass(frozen=True)
class MarkPrice:
    time_ms: int
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_time_ms: int


@dataclass(frozen=True)
class Liquidation:
    time_ms: int
    symbol: str
    side: str  # SELL = a long position was liquidated, BUY = a short
    price: float
    avg_price: float
    qty: float
    status: str

    @property
    def quote(self) -> float:
        return (self.avg_price or self.price) * self.qty


def parse_agg_trade(data: dict[str, Any]) -> Trade:
    """Spot and futures aggTrade share the same fields."""
    return Trade(
        time_ms=int(data["T"]),
        price=float(data["p"]),
        qty=float(data["q"]),
        buyer_is_maker=bool(data["m"]),
    )


def _levels(raw: list[list[str]]) -> list[tuple[float, float]]:
    return [(float(price), float(qty)) for price, qty in raw]


def parse_depth(data: dict[str, Any]) -> BookSnapshot:
    """Spot partial depth uses bids/asks; futures depthUpdate uses b/a."""
    bids = data.get("bids", data.get("b", []))
    asks = data.get("asks", data.get("a", []))
    return BookSnapshot(bids=_levels(bids), asks=_levels(asks))


def depth_event_time(data: dict[str, Any]) -> int | None:
    """Futures depth carries an event time; spot partial depth does not."""
    value = data.get("E")
    return int(value) if value is not None else None


def parse_mark_price(data: dict[str, Any]) -> MarkPrice:
    return MarkPrice(
        time_ms=int(data["E"]),
        mark_price=float(data["p"]),
        index_price=float(data["i"]),
        funding_rate=float(data["r"]),
        next_funding_time_ms=int(data["T"]),
    )


def parse_liquidation(data: dict[str, Any]) -> Liquidation:
    order = data["o"]
    return Liquidation(
        time_ms=int(order["T"]),
        symbol=str(order["s"]),
        side=str(order["S"]),
        price=float(order["p"]),
        avg_price=float(order.get("ap") or 0.0),
        qty=float(order.get("z") or order["q"]),
        status=str(order.get("X", "")),
    )
