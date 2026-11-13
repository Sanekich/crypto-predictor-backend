"""Continuous collector of Binance real-time data for one symbol.

Connections (Binance routes as of 2026):
- spot:            aggTrade + order book (top 20)        stream.binance.com
- futures market:  aggTrade + mark price/funding + liquidations   fstream.binance.com/market
- futures public:  order book (top 20)                   fstream.binance.com/public
- futures REST:    open interest, polled every minute

Every connection reconnects automatically with exponential backoff (Binance
also closes connections after 24 hours). Connects and disconnects are logged
to the `connection_events` dataset so gaps in the data are documented.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd

from app.data.live.aggregators import (
    MarkPriceAggregator,
    OrderBookAggregator,
    TradeFlowAggregator,
)
from app.data.live.messages import (
    depth_event_time,
    parse_agg_trade,
    parse_depth,
    parse_liquidation,
    parse_mark_price,
)
from app.data.live.writer import LiveDataWriter

logger = logging.getLogger(__name__)

SPOT_WS = "wss://stream.binance.com:9443/stream"
FUTURES_MARKET_WS = "wss://fstream.binance.com/market/stream"
FUTURES_PUBLIC_WS = "wss://fstream.binance.com/public/stream"
FUTURES_REST = "https://fapi.binance.com"

# Datasets written to data/live/<dataset>/
TRADES_SPOT = "trades_spot"
TRADES_FUTURES = "trades_futures"
BOOK_SPOT = "book_spot"
BOOK_FUTURES = "book_futures"
MARK_PRICE = "mark_price"
LIQUIDATIONS = "liquidations"
OPEN_INTEREST = "open_interest"
CONNECTION_EVENTS = "connection_events"


@dataclass
class Connection:
    name: str
    url: str
    handlers: dict[str, Callable[[dict[str, Any]], None]]
    aggregators: list[Any] = field(default_factory=list)


def _now_ms() -> int:
    return int(time.time() * 1000)


class LiveCollector:
    def __init__(
        self,
        writer: LiveDataWriter,
        symbol: str = "BTCUSDT",
        flush_interval: float = 60.0,
        oi_interval: float = 60.0,
        clock_ms: Callable[[], int] = _now_ms,
    ) -> None:
        self.writer = writer
        self.symbol = symbol.upper()
        self.flush_interval = flush_interval
        self.oi_interval = oi_interval
        self._clock_ms = clock_ms
        s = self.symbol.lower()

        self.trades_spot = TradeFlowAggregator()
        self.trades_futures = TradeFlowAggregator()
        self.book_spot = OrderBookAggregator()
        self.book_futures = OrderBookAggregator()
        self.mark = MarkPriceAggregator()

        self.connections = [
            Connection(
                "spot",
                f"{SPOT_WS}?streams={s}@aggTrade/{s}@depth20@1000ms",
                {
                    f"{s}@aggTrade": lambda d: self._add(TRADES_SPOT, self.trades_spot.add(parse_agg_trade(d))),
                    # Spot partial depth has no timestamp: use the local receive time
                    f"{s}@depth20@1000ms": lambda d: self._add(
                        BOOK_SPOT, self.book_spot.add(parse_depth(d), self._clock_ms())
                    ),
                },
                [self.trades_spot, self.book_spot],
            ),
            Connection(
                "futures_market",
                f"{FUTURES_MARKET_WS}?streams={s}@aggTrade/{s}@markPrice@1s/!forceOrder@arr",
                {
                    f"{s}@aggTrade": lambda d: self._add(TRADES_FUTURES, self.trades_futures.add(parse_agg_trade(d))),
                    f"{s}@markPrice@1s": lambda d: self._add(MARK_PRICE, self.mark.add(parse_mark_price(d))),
                    "!forceOrder@arr": self._on_liquidation,
                },
                [self.trades_futures, self.mark],
            ),
            Connection(
                "futures_public",
                f"{FUTURES_PUBLIC_WS}?streams={s}@depth20@500ms",
                {
                    f"{s}@depth20@500ms": lambda d: self._add(
                        BOOK_FUTURES,
                        self.book_futures.add(parse_depth(d), depth_event_time(d) or self._clock_ms()),
                    ),
                },
                [self.book_futures],
            ),
        ]
        self.messages: dict[str, int] = {c.name: 0 for c in self.connections}
        self.last_message_ms: dict[str, int | None] = {c.name: None for c in self.connections}
        self.connected: dict[str, bool] = {c.name: False for c in self.connections}
        self.errors = 0
        self.flush_errors = 0
        self.last_flush_error: str | None = None
        self.started_ms = clock_ms()

    # --- message handling (synchronous, unit-testable) -------------------------

    def _add(self, dataset: str, rows: list[dict[str, Any]]) -> None:
        self.writer.add(dataset, rows)

    def _on_liquidation(self, data: dict[str, Any]) -> None:
        liquidation = parse_liquidation(data)
        if liquidation.symbol != self.symbol:
            return
        self.writer.add(
            LIQUIDATIONS,
            [
                {
                    "timestamp": pd.Timestamp(liquidation.time_ms, unit="ms", tz="UTC"),
                    "side": liquidation.side,
                    "price": liquidation.price,
                    "avg_price": liquidation.avg_price,
                    "qty": liquidation.qty,
                    "quote": liquidation.quote,
                    "status": liquidation.status,
                }
            ],
        )

    def handle_message(self, connection: Connection, raw: str | bytes) -> None:
        self.messages[connection.name] += 1
        self.last_message_ms[connection.name] = self._clock_ms()
        try:
            message = json.loads(raw)
            handler = connection.handlers.get(message.get("stream", ""))
            if handler is not None:
                handler(message["data"])
        except (KeyError, ValueError, TypeError) as exc:
            self.errors += 1
            logger.warning("Bad message on %s: %s", connection.name, exc)

    def log_connection_event(self, connection: Connection, event: str, detail: str = "") -> None:
        self.connected[connection.name] = event == "connected"
        self.writer.add(
            CONNECTION_EVENTS,
            [
                {
                    # unique per event: millisecond timestamp
                    "timestamp": pd.Timestamp(self._clock_ms(), unit="ms", tz="UTC"),
                    "connection": connection.name,
                    "event": event,
                    "detail": detail[:300],
                }
            ],
        )

    def on_disconnect(self, connection: Connection) -> None:
        """Partially observed minutes would be wrong: drop them."""
        for aggregator in connection.aggregators:
            aggregator.reset()

    def status(self) -> dict[str, Any]:
        now = self._clock_ms()
        return {
            "symbol": self.symbol,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "uptime_minutes": round((now - self.started_ms) / 60_000, 1),
            "connections": {
                name: {
                    "connected": self.connected[name],
                    "messages": self.messages[name],
                    "seconds_since_last_message": None
                    if self.last_message_ms[name] is None
                    else round((now - self.last_message_ms[name]) / 1000, 1),
                }
                for name in self.messages
            },
            "rows_written": dict(self.writer.rows_written),
            "parse_errors": self.errors,
            "flush_errors": self.flush_errors,
            "last_flush_error": self.last_flush_error,
        }

    # --- async runtime ------------------------------------------------------------

    async def run_connection(
        self,
        connection: Connection,
        connect: Callable[[str], Any] | None = None,
        max_backoff: float = 60.0,
    ) -> None:
        import websockets

        connect = connect or (lambda url: websockets.connect(url, open_timeout=15, ping_interval=20, max_size=2**22))
        backoff = 1.0
        try:
            while True:
                try:
                    async with connect(connection.url) as ws:
                        logger.info("Connected: %s", connection.name)
                        self.log_connection_event(connection, "connected")
                        backoff = 1.0
                        async for raw in ws:
                            self.handle_message(connection, raw)
                    detail = "closed by server"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - any network error -> reconnect
                    detail = f"{type(exc).__name__}: {exc}"
                logger.warning("Disconnected: %s (%s), reconnecting in %.0fs", connection.name, detail, backoff)
                self.log_connection_event(connection, "disconnected", detail)
                self.on_disconnect(connection)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
        except asyncio.CancelledError:
            self.log_connection_event(connection, "stopped")
            raise

    async def poll_open_interest(self, client: httpx.AsyncClient | None = None) -> None:
        own_client = client is None
        client = client or httpx.AsyncClient(base_url=FUTURES_REST, timeout=10)
        try:
            while True:
                try:
                    response = await client.get("/fapi/v1/openInterest", params={"symbol": self.symbol})
                    response.raise_for_status()
                    data = response.json()
                    self.writer.add(
                        OPEN_INTEREST,
                        [
                            {
                                "timestamp": pd.Timestamp(int(data["time"]), unit="ms", tz="UTC"),
                                "open_interest": float(data["openInterest"]),
                            }
                        ],
                    )
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    logger.warning("Open interest poll failed: %s", exc)
                await asyncio.sleep(self.oi_interval)
        finally:
            if own_client:
                await client.aclose()

    async def flush_periodically(self) -> None:
        while True:
            await asyncio.sleep(self.flush_interval)
            self._safe_flush()

    def flush(self) -> None:
        rows = self.writer.flush()
        self.writer.write_status(self.status())
        logger.info(
            "Flushed %d rows | messages %s",
            rows,
            ", ".join(f"{k}={v}" for k, v in self.messages.items()),
        )

    def _safe_flush(self) -> None:
        """Flush without allowing one bad row to kill the periodic task."""
        try:
            self.last_flush_error = None
            self.flush()
        except Exception as exc:  # noqa: BLE001 - keep collection alive on disk errors
            self.flush_errors += 1
            self.last_flush_error = f"{type(exc).__name__}: {exc}"[:500]
            logger.exception("Live-data flush failed; will retry on the next interval")
            try:
                self.writer.write_status(self.status())
            except Exception:  # noqa: BLE001 - status is best effort during an I/O failure
                logger.exception("Could not write collector error status")

    async def run(self, extra_tasks: list[Awaitable[None]] | None = None) -> None:
        # Publish an immediate status so the dashboard does not show an old run
        # for the first flush interval.
        self._safe_flush()
        tasks = [asyncio.create_task(self.run_connection(c)) for c in self.connections]
        tasks.append(asyncio.create_task(self.poll_open_interest()))
        tasks.append(asyncio.create_task(self.flush_periodically()))
        for extra in extra_tasks or []:
            tasks.append(asyncio.ensure_future(extra))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._safe_flush()
            logger.info("Collector stopped, buffers flushed")
