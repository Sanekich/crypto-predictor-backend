"""Run the API and live collector together with collector restart supervision.

Use this as a simple single-service deployment entrypoint:
    python scripts/start_backend_with_collector.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(*args: str) -> list[str]:
    return [sys.executable, *args]


def main() -> int:
    symbol = os.getenv("COLLECTOR_SYMBOL", "BTCUSDT")
    port = os.getenv("PORT", "8000")
    api = subprocess.Popen(command("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", port), cwd=ROOT)
    collector: subprocess.Popen[bytes] | None = None
    stopping = False

    def stop(*_args: object) -> None:
        nonlocal stopping
        stopping = True
        for process in (collector, api):
            if process is not None and process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        while api.poll() is None and not stopping:
            if collector is None or collector.poll() is not None:
                if collector is not None:
                    print(f"Collector exited with code {collector.returncode}; restarting in 5 seconds", flush=True)
                    time.sleep(5)
                collector = subprocess.Popen(
                    command("scripts/collect_live.py", "--symbol", symbol),
                    cwd=ROOT,
                )
            time.sleep(2)
    finally:
        stop()
        for process in (collector, api):
            if process is not None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
    return api.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
