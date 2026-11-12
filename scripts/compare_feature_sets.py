"""Stages 2-3: do derivatives features improve the models? Walk-forward comparison.

Every feature set is evaluated on exactly the same rows (the period where all
sources exist) and the same expanding folds. A set counts as an improvement only
if it beats the base set on average AND in most folds.

    python scripts/compare_feature_sets.py
    python scripts/compare_feature_sets.py --folds 5 --direction-horizons 4h 24h --range-horizons 24h 7d

Report: data/reports/feature_comparison_<date>.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.data.derivatives.store import DerivativesStore  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402
from app.features.derivatives import GROUP_DATASET  # noqa: E402
from app.features.feature_pipeline import FeatureConfig, build_dataset  # noqa: E402
from app.ml.evaluation.walk_forward import evaluate_direction_walk_forward, evaluate_range_walk_forward  # noqa: E402

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "v1 base (candles)": (),
    "v2 +funding +basis": ("funding", "basis"),
    "v2b +dvol": ("funding", "basis", "dvol"),
    "v3 +open interest": ("funding", "basis", "open_interest"),
    "v3 all": ("funding", "basis", "open_interest", "dvol"),
}


def common_frames(ohlcv, derivatives, horizon, sets):
    """Datasets of all sets restricted to timestamps present in every one of them."""
    datasets = {
        name: build_dataset(ohlcv, "1h", horizon, FeatureConfig(derivatives=groups), derivatives)
        for name, groups in sets.items()
    }
    common = None
    for ds in datasets.values():
        stamps = set(ds.frame["timestamp"])
        common = stamps if common is None else common & stamps
    return {
        name: (ds, ds.frame[ds.frame["timestamp"].isin(common)].reset_index(drop=True))
        for name, ds in datasets.items()
    }


def summarize_direction(folds: list[dict]) -> dict:
    skills = [1 - f["brier"] / f["brier_base_rate"] for f in folds]
    signals = [f["signals"]["0.60"] for f in folds]
    n = sum(s["independent_signals"] for s in signals)
    hits = sum((s["independent_hit_rate"] or 0) * s["independent_signals"] for s in signals)
    return {
        "brier_skill_mean": float(np.mean(skills)),
        "brier_skill_by_fold": skills,
        "accuracy_mean": float(np.mean([f["accuracy"] for f in folds])),
        "majority_baseline_mean": float(np.mean([f["majority_baseline_accuracy"] for f in folds])),
        "max_confidence": float(max(f["max_confidence"] for f in folds)),
        "signals_60_independent": int(n),
        "signals_60_hit_rate": hits / n if n else None,
    }


def summarize_range(folds: list[dict]) -> dict:
    return {
        "pinball_skill_mean": float(np.mean([f["pinball_skill_vs_constant"] for f in folds])),
        "pinball_skill_by_fold": [f["pinball_skill_vs_constant"] for f in folds],
        "coverage_mean": float(np.mean([f["coverage"] for f in folds])),
        "width_correlation_mean": float(np.mean([f["width_correlation"] or 0 for f in folds])),
        "mean_width": float(np.mean([f["mean_width"] for f in folds])),
    }


def verdict(base_scores: list[float], scores: list[float]) -> str:
    better = sum(s > b for s, b in zip(scores, base_scores))
    if np.mean(scores) > np.mean(base_scores) and better >= (len(scores) + 1) // 2 + 1:
        return f"BETTER ({better}/{len(scores)} folds)"
    if np.mean(scores) > np.mean(base_scores):
        return f"unclear ({better}/{len(scores)} folds)"
    return f"no gain ({better}/{len(scores)} folds)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--direction-horizons", nargs="*", default=["4h", "24h"])
    parser.add_argument("--range-horizons", nargs="*", default=["24h", "7d"])
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging("WARNING")
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    symbol = args.symbol.upper()
    ohlcv = MarketDataStorage(data_dir).load_processed(settings.market_data_provider, symbol, "1h")
    derivatives = DerivativesStore(data_dir).load_all(symbol)

    sets = {
        name: groups
        for name, groups in FEATURE_SETS.items()
        if all(GROUP_DATASET[g] in derivatives for g in groups)
    }
    skipped = sorted(set(FEATURE_SETS) - set(sets))
    if skipped:
        print(f"Skipped (data not downloaded): {', '.join(skipped)}")

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "folds": args.folds, "tasks": {}}
    base_name = "v1 base (candles)"

    tasks = [("direction", h) for h in args.direction_horizons] + [("range", h) for h in args.range_horizons]
    for kind, horizon in tasks:
        frames = common_frames(ohlcv, derivatives, horizon, sets)
        any_ds, any_frame = next(iter(frames.values()))
        period = f"{any_frame['timestamp'].iloc[0]:%Y-%m-%d} .. {any_frame['timestamp'].iloc[-1]:%Y-%m-%d}"
        print(f"\n=== {kind} {horizon} | {len(any_frame)} common rows | {period}", flush=True)

        results = {}
        for name, (ds, frame) in frames.items():
            if kind == "direction":
                folds = evaluate_direction_walk_forward(
                    frame, ds.feature_columns, ds.target_direction_column, ds.horizon_periods, args.folds
                )
                results[name] = {"features": len(ds.feature_columns), **summarize_direction(folds)}
            else:
                folds = evaluate_range_walk_forward(
                    frame, ds.feature_columns, ds.target_return_column, ds.horizon_periods, args.folds
                )
                results[name] = {"features": len(ds.feature_columns), **summarize_range(folds)}

        key = "brier_skill_by_fold" if kind == "direction" else "pinball_skill_by_fold"
        for name, r in results.items():
            r["verdict_vs_base"] = "-" if name == base_name else verdict(results[base_name][key], r[key])
            if kind == "direction":
                hit = r["signals_60_hit_rate"]
                print(
                    f"{name:<20} skill {r['brier_skill_mean']:+.4f}  acc {r['accuracy_mean']:.1%} "
                    f"(majority {r['majority_baseline_mean']:.1%})  max conf {r['max_confidence']:.1%}  "
                    f">=60%: {r['signals_60_independent']} sig, hit {'n/a' if hit is None else f'{hit:.1%}'}  "
                    f"{r['verdict_vs_base']}",
                    flush=True,
                )
            else:
                print(
                    f"{name:<20} pinball skill {r['pinball_skill_mean']:+.1%}  coverage {r['coverage_mean']:.1%}  "
                    f"width {r['mean_width']:.2%}  width~|ret| {r['width_correlation_mean']:.3f}  {r['verdict_vs_base']}",
                    flush=True,
                )
        report["tasks"][f"{kind}_{horizon}"] = {"rows": len(any_frame), "period": period, "results": results}

    out = data_dir / "reports" / f"feature_comparison_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
