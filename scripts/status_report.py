"""Print a compact health report for local data, live collection, and models.

Run from the backend repository root:
    python scripts/status_report.py

The command is intentionally read-only. It does not retrain models or modify files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR {path}: {exc}", file=sys.stderr)
        return None


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def model_metadata(artifacts_dir: Path) -> list[dict[str, Any]]:
    models_dir = artifacts_dir / "models"
    result: list[dict[str, Any]] = []
    if not models_dir.exists():
        return result
    for path in models_dir.iterdir():
        metadata = load_json(path / "metadata.json")
        if metadata:
            result.append(metadata)
    return sorted(result, key=lambda item: item.get("trained_at", ""), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    report_path = args.data_dir / "processed" / "binance" / f"{args.symbol}_{args.interval}.report.json"
    live_path = args.data_dir / "live" / f"_status_{args.symbol}.json"
    report = load_json(report_path)
    live = load_json(live_path)
    models = model_metadata(args.artifacts_dir)

    print(f"CryptoPredictor status: {args.symbol} {args.interval}")
    print("=" * 56)

    exit_code = 0
    if report is None:
        print(f"DATA: missing report ({report_path})")
        exit_code = 1
    else:
        valid = bool(report.get("is_valid"))
        print(f"DATA: {'OK' if valid else 'INVALID'}")
        print(f"  rows: {report.get('rows', 'n/a')}")
        print(f"  period: {report.get('first_timestamp', 'n/a')} -> {report.get('last_timestamp', 'n/a')}")
        print(f"  gaps: {report.get('gap_count', 0)} ({report.get('missing_candles', 0)} candles)")
        if not valid:
            exit_code = 1
            for error in report.get("errors", []):
                print(f"  error: {error}")

    if live is None:
        print("LIVE: no status file")
    else:
        updated = parse_time(live.get("updated_at"))
        age = "n/a"
        age_minutes: float | None = None
        if updated is not None:
            age_minutes = (datetime.now(timezone.utc) - updated).total_seconds() / 60
            age = f"{age_minutes:.1f} min ago"
        connections = live.get("connections", {})
        connected = all(item.get("connected", False) for item in connections.values()) if connections else False
        stale = age_minutes is not None and age_minutes > 3 * 60
        live_ok = connected and not stale and live.get("parse_errors", 0) == 0
        print(f"LIVE: {'OK' if live_ok else 'CHECK'} (updated {age})")
        connection_text = ", ".join(
            f"{name}={info.get('connected', False)}" for name, info in connections.items()
        ) or "n/a"
        print(f"  connections: {connection_text}")
        print(f"  parse errors: {live.get('parse_errors', 'n/a')}")
        if stale:
            print("  warning: live status is older than 3 hours")

    print(f"MODELS: {len(models)} saved")
    for meta in models:
        model_type = meta.get("model_type", "unknown")
        horizon = meta.get("horizon", "n/a")
        status = meta.get("status", "n/a")
        model_id = meta.get("model_id", "unknown")
        print(f"  {model_type} {horizon}: {status} ({model_id})")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
