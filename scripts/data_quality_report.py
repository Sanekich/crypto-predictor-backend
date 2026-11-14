"""Human-readable read-only quality report for CryptoPredictor market data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


UTC = "UTC"


def utc_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], utc=True, errors="coerce")


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def check_ohlcv(path: Path) -> tuple[dict, list[str]]:
    frame = pd.read_parquet(path)
    ts = utc_series(frame, "timestamp")
    expected = pd.date_range(ts.min(), ts.max(), freq="h", tz=UTC)
    missing = expected.difference(ts.dropna().sort_values())
    issues: list[str] = []
    checks = {
        "rows": len(frame),
        "first": ts.min(),
        "last": ts.max(),
        "duplicates": int(ts.duplicated().sum()),
        "null_timestamps": int(ts.isna().sum()),
        "sorted": bool(ts.is_monotonic_increasing),
        "missing_candles": len(missing),
        "bad_ohlc": int((frame.high < frame[["open", "close"]].max(axis=1)).sum()
                        + (frame.low > frame[["open", "close"]].min(axis=1)).sum()),
        "nonpositive_prices": int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "negative_volume": int((frame["volume"] < 0).sum()),
    }
    if checks["duplicates"] or checks["null_timestamps"] or checks["bad_ohlc"] or checks["nonpositive_prices"] or checks["negative_volume"]:
        issues.append("OHLCV contains structural errors")
    if not checks["sorted"]:
        issues.append("OHLCV timestamps are not sorted")
    if checks["missing_candles"]:
        issues.append(f"{checks['missing_candles']} missing hourly candles")
    return checks, issues


def check_derivative(path: Path, name: str, candle_last: pd.Timestamp) -> tuple[dict, list[str]]:
    frame = pd.read_parquet(path)
    available = utc_series(frame, "available_at")
    issues: list[str] = []
    checks = {
        "rows": len(frame),
        "first_available": available.min(),
        "last_available": available.max(),
        "age_hours_vs_candles": round((candle_last - available.max()).total_seconds() / 3600, 2),
        "duplicates": int(available.duplicated().sum()),
        "null_available_at": int(available.isna().sum()),
        "sorted": bool(available.is_monotonic_increasing),
        "null_values": int(frame.drop(columns=["available_at"], errors="ignore").isna().sum().sum()),
    }
    if "timestamp" in frame:
        timestamp = utc_series(frame, "timestamp")
        checks["negative_availability_delay"] = int(((available - timestamp) < pd.Timedelta(0)).sum())
        if checks["negative_availability_delay"]:
            issues.append("some values have available_at before their event timestamp")
    if checks["duplicates"] or checks["null_available_at"]:
        issues.append("derivative timestamps are duplicated or missing")
    if checks["null_values"]:
        issues.append(f"{checks['null_values']} optional metric values are null")
    if not checks["sorted"]:
        issues.append("timestamps are not sorted")
    if name == "metrics" and checks["age_hours_vs_candles"] > 48:
        issues.append("metrics history is more than 48h behind candles")
    elif name != "metrics" and checks["age_hours_vs_candles"] > 24:
        issues.append("derivative history is more than 24h behind candles")
    return checks, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only CryptoPredictor data quality report")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    root = Path(args.data_dir)
    candle_path = root / "processed" / "binance" / f"{args.symbol}_1h.parquet"
    if not candle_path.exists():
        print(f"BLOCKER: missing OHLCV file: {candle_path}", file=sys.stderr)
        return 2

    result: dict = {"symbol": args.symbol, "ohlcv": {}, "derivatives": {}, "live": {}, "issues": []}
    ohlcv, issues = check_ohlcv(candle_path)
    result["ohlcv"] = ohlcv
    result["issues"].extend(issues)

    for name in ("funding", "premium", "metrics", "dvol"):
        path = root / "derivatives" / name / f"{args.symbol}.parquet"
        if not path.exists():
            result["derivatives"][name] = {"missing_file": str(path)}
            result["issues"].append(f"missing derivative file: {name}")
            continue
        checks, derivative_issues = check_derivative(path, name, ohlcv["last"])
        result["derivatives"][name] = checks
        result["issues"].extend(f"{name}: {issue}" for issue in derivative_issues)

    status_path = root / "live" / f"_status_{args.symbol}.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        updated = pd.Timestamp(status.get("updated_at"))
        age_minutes = (pd.Timestamp.now(tz=UTC) - updated).total_seconds() / 60
        result["live"] = {
            "updated_at": updated,
            "age_minutes": round(age_minutes, 1),
            "parse_errors": status.get("parse_errors", 0),
            "flush_errors": status.get("flush_errors", 0),
            "last_flush_error": status.get("last_flush_error"),
            "connections": status.get("connections", {}),
        }
        if age_minutes > 180:
            result["issues"].append(f"live status is stale ({round(age_minutes / 60, 1)}h)")
        if status.get("parse_errors", 0) or status.get("flush_errors", 0):
            result["issues"].append("live collector reports parse or flush errors")
    else:
        result["issues"].append("live collector status file is missing")

    structural = any(
        "structural errors" in issue
        or "missing derivative file" in issue
        or "timestamps are duplicated or missing" in issue
        or "available_at before" in issue
        for issue in result["issues"]
    )
    result["training_ready"] = not structural and bool(ohlcv["rows"])
    result["live_ready"] = not any("live status" in issue or "collector status" in issue or "collector reports" in issue for issue in result["issues"])
    result["quality"] = "GOOD" if not result["issues"] else ("USABLE_WITH_WARNINGS" if not structural else "INVALID")

    if args.as_json:
        print(json.dumps(result, default=fmt, indent=2))
        return 1 if structural else 0

    print(f"CryptoPredictor data quality: {args.symbol} 1h")
    print("=" * 56)
    print(f"QUALITY: {result['quality']}")
    print(f"TRAINING DATA: {'READY' if result['training_ready'] else 'BLOCKED'}")
    print(f"OHLCV: {ohlcv['rows']} rows | {fmt(ohlcv['first'])} -> {fmt(ohlcv['last'])}")
    print(f"  duplicates={ohlcv['duplicates']} missing_candles={ohlcv['missing_candles']} bad_ohlc={ohlcv['bad_ohlc']}")
    for name, checks in result["derivatives"].items():
        if "missing_file" in checks:
            print(f"{name}: MISSING")
        else:
            print(f"{name}: {checks['rows']} rows | latest={fmt(checks['last_available'])} | age_vs_candle={checks['age_hours_vs_candles']}h")
            print(f"  duplicates={checks['duplicates']} nulls={checks['null_values']} sorted={checks['sorted']}")
    live = result["live"]
    print(f"LIVE: {'READY' if result['live_ready'] else 'STALE/ERROR'} | age={live.get('age_minutes', 'n/a')} min | parse_errors={live.get('parse_errors', 'n/a')} | flush_errors={live.get('flush_errors', 'n/a')}")
    if result["issues"]:
        print("ISSUES:")
        for issue in result["issues"]:
            print(f"- {issue}")
    else:
        print("ISSUES: none")
    return 1 if structural else 0


if __name__ == "__main__":
    raise SystemExit(main())
