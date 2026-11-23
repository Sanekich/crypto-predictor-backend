"""Euphoria / panic index: event study ("does the crowd's extreme predict a reversal?").

    python scripts/download_derivatives.py      # funding, open interest, Fear & Greed
    python scripts/euphoria_study.py

Prints the hit rate of the reversal hypothesis per zone, threshold and horizon,
and writes data/reports/euphoria_<date>.json (shown on the site, /euphoria).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.data.derivatives.store import DerivativesStore  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402
from app.ml.euphoria.index import build_euphoria_frame  # noqa: E402
from app.ml.euphoria.study import run_study  # noqa: E402


def pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.1%}"


def main() -> int:
    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    ohlcv = MarketDataStorage(data_dir).load_processed(settings.market_data_provider, "BTCUSDT", "1h")
    if ohlcv is None:
        print("No processed candles. Run scripts/download_data.py first.", file=sys.stderr)
        return 1
    try:
        frame = build_euphoria_frame(ohlcv, DerivativesStore(data_dir).load_all("BTCUSDT"))
    except ValueError as exc:
        print(f"{exc}. Run scripts/download_derivatives.py first.", file=sys.stderr)
        return 1

    study = run_study(frame)
    print(f"Euphoria index: {study['period']['start'][:10]} .. {study['period']['end'][:10]}, {study['hours_with_index']} hours")
    print(f"{'zone':<9}{'expect':<7}{'thr':>5}{'horizon':>8}{'events':>8}{'hit':>8}{'base':>8}{'lower95':>9}{'mean ret':>10}{'years +':>9}")
    for r in study["results"]:
        mean = "     n/a" if r["mean_return"] is None else f"{r['mean_return']:+9.2%}"
        print(f"{r['zone']:<9}{r['expected']:<7}{r['threshold']:>5.1f}{r['horizon']:>8}{r['events']:>8}{pct(r['hit_rate']):>8}"
              f"{pct(r['base_rate']):>8}{pct(r['hit_rate_lower_95']):>9}{mean:>10}{r['positive_years']:>5}/{r['years']}")

    out = data_dir / "reports" / f"euphoria_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), **study}, indent=2), encoding="utf-8")
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
