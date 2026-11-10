"""Client for Binance public market data (no API key required)."""

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from app.data.intervals import interval_to_ms

logger = logging.getLogger(__name__)

Kline = list[Any]


class BinanceAPIError(RuntimeError):
    """Raised when Binance returns an error or cannot be reached."""


class BinanceClient:
    """Fetches OHLCV candles (klines) from the Binance spot REST API.

    Handles pagination, rate limiting (HTTP 429 with Retry-After) and
    transient network/server errors with exponential backoff.
    """

    MAX_LIMIT = 1000

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout: float = 10.0,
        max_retries: int = 5,
        request_pause: float = 0.2,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._max_retries = max_retries
        self._request_pause = request_pause
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        for attempt in range(self._max_retries + 1):
            is_last = attempt == self._max_retries
            backoff = 2**attempt

            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                if is_last:
                    raise BinanceAPIError(f"Network error: {exc}") from exc
                logger.warning("Network error (%s), retrying in %ss", exc, backoff)
                self._sleep(backoff)
                continue

            status = response.status_code
            if status == 200:
                return response.json()

            if status == 418:
                raise BinanceAPIError(
                    "Binance temporarily banned this IP for exceeding rate limits (HTTP 418)"
                )

            if status == 429 and not is_last:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning("Rate limit hit (HTTP 429), waiting %ss", retry_after)
                self._sleep(retry_after)
                continue

            if status >= 500 and not is_last:
                logger.warning("Binance server error %s, retrying in %ss", status, backoff)
                self._sleep(backoff)
                continue

            raise BinanceAPIError(f"Binance API error {status}: {response.text[:300]}")

        raise BinanceAPIError("Exceeded maximum number of retries")  # pragma: no cover

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = MAX_LIMIT,
    ) -> list[Kline]:
        interval_to_ms(interval)  # validates interval
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, self.MAX_LIMIT),
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        return self._get("/api/v3/klines", params)

    def iter_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int | None = None,
    ) -> Iterator[list[Kline]]:
        """Yield batches of klines from start_ms until end_ms (or the present)."""
        step = interval_to_ms(interval)
        cursor = start_ms

        while True:
            batch = self.get_klines(symbol, interval, start_ms=cursor, end_ms=end_ms)
            if not batch:
                return
            yield batch

            cursor = int(batch[-1][0]) + step
            if len(batch) < self.MAX_LIMIT:
                return
            if end_ms is not None and cursor > end_ms:
                return
            self._sleep(self._request_pause)
