import httpx
import pytest

from app.data.collection.binance_client import BinanceAPIError, BinanceClient
from tests.factories import START_MS, make_klines

DATASET = make_klines(2500)


def klines_handler(request: httpx.Request) -> httpx.Response:
    params = request.url.params
    start = int(params.get("startTime", 0))
    end = int(params["endTime"]) if "endTime" in params else None
    limit = int(params["limit"])
    rows = [k for k in DATASET if k[0] >= start and (end is None or k[0] <= end)]
    return httpx.Response(200, json=rows[:limit])


def make_client(handler, sleeps: list[float] | None = None) -> BinanceClient:
    sleeps = sleeps if sleeps is not None else []
    return BinanceClient(
        base_url="https://test.local",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )


def test_iter_klines_paginates_full_history() -> None:
    client = make_client(klines_handler)
    batches = list(client.iter_klines("btcusdt", "1h", start_ms=START_MS))

    assert [len(b) for b in batches] == [1000, 1000, 500]
    open_times = [k[0] for batch in batches for k in batch]
    assert open_times == [k[0] for k in DATASET]


def test_iter_klines_respects_end() -> None:
    client = make_client(klines_handler)
    end_ms = DATASET[9][0]
    rows = [k for batch in client.iter_klines("BTCUSDT", "1h", START_MS, end_ms) for k in batch]
    assert len(rows) == 10


def test_symbol_is_uppercased() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["symbol"])
        return httpx.Response(200, json=[])

    make_client(handler).get_klines("btcusdt", "1h")
    assert seen == ["BTCUSDT"]


def test_retries_after_rate_limit() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return klines_handler(request)

    sleeps: list[float] = []
    rows = make_client(handler, sleeps).get_klines("BTCUSDT", "1h", start_ms=START_MS, limit=5)

    assert len(rows) == 5
    assert sleeps == [7]


def test_retries_server_errors_then_fails() -> None:
    sleeps: list[float] = []
    client = make_client(lambda request: httpx.Response(503), sleeps)
    with pytest.raises(BinanceAPIError):
        client.get_klines("BTCUSDT", "1h")
    assert sleeps == [1, 2, 4, 8, 16]


def test_ip_ban_is_not_retried() -> None:
    sleeps: list[float] = []
    client = make_client(lambda request: httpx.Response(418), sleeps)
    with pytest.raises(BinanceAPIError, match="418"):
        client.get_klines("BTCUSDT", "1h")
    assert sleeps == []


def test_client_error_is_raised() -> None:
    client = make_client(lambda request: httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."}))
    with pytest.raises(BinanceAPIError, match="Invalid symbol"):
        client.get_klines("NOPE", "1h")


def test_invalid_interval_rejected() -> None:
    with pytest.raises(ValueError):
        make_client(klines_handler).get_klines("BTCUSDT", "7h")
