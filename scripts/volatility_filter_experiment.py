"""Does the volatility model make direction signals more accurate? (combination test)

Direction model: the old-logic 4h model ("higher or lower after 4 hours"), the
only short horizon with stable skill. Volatility model: "move of +-1% within 4h".
On each expanding walk-forward fold both are trained on the same past data; on
the unseen test period the direction accuracy is compared:

- on all hours;
- only on hours where the volatility model expects a big move (top 50 / 25 / 10%
  of P(move), thresholds taken from the training period - no peeking);
- for confident direction signals (>= 55% / 60%) with and without the filter.

Hit rates are counted on independent signals (at least 4 hours apart).

    python scripts/volatility_filter_experiment.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.features.feature_pipeline import build_dataset  # noqa: E402
from app.ml.barrier.dataset import load_market_data  # noqa: E402
from app.ml.evaluation.metrics import independent_positions, wilson_lower_bound  # noqa: E402
from app.ml.evaluation.walk_forward import expanding_folds  # noqa: E402
from app.ml.models.direction_model import XGBoostDirectionModel  # noqa: E402
from app.ml.volatility.configs import VOLATILITY_CONFIGS  # noqa: E402
from app.ml.volatility.dataset import build_volatility_dataset  # noqa: E402

TOP_SHARES = (1.0, 0.5, 0.25, 0.10)
CONFIDENCE = (0.50, 0.55, 0.60)


def fit(X, y, split) -> XGBoostDirectionModel:
    model = XGBoostDirectionModel().fit(X[split.train], y[split.train], X[split.early_stopping], y[split.early_stopping])
    return model.calibrate(X[split.calibration], y[split.calibration])


def main() -> int:
    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    ohlcv, _ = load_market_data(data_dir, settings.market_data_provider, "BTCUSDT", "1h", "candles")

    direction = build_dataset(ohlcv, "1h", "4h")
    vol = build_volatility_dataset(ohlcv, "1h", VOLATILITY_CONFIGS["move_1pct_4h"], "candles")
    frame = direction.frame.merge(vol.frame[["timestamp", "label"]].rename(columns={"label": "big_move"}), on="timestamp")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    X = frame[direction.feature_columns].to_numpy(dtype="float64")
    y_dir = frame[direction.target_direction_column].to_numpy(dtype="float64")
    y_vol = frame["big_move"].to_numpy(dtype="float64")
    ret = frame[direction.target_return_column].to_numpy(dtype="float64")
    h = direction.horizon_periods

    # pooled counters: (share, confidence) -> [independent n, hits, sum |return|]
    pooled = {(s, c): [0, 0, 0.0] for s in TOP_SHARES for c in CONFIDENCE}
    for split in expanding_folds(frame["timestamp"], h, folds=5):
        dir_model = fit(X, y_dir, split)
        vol_model = fit(X, y_vol, split)
        p_move_cal = vol_model.predict_proba_up(X[split.calibration])
        p_up = dir_model.predict_proba_up(X[split.test])
        p_move = vol_model.predict_proba_up(X[split.test])
        correct = (p_up >= 0.5) == (y_dir[split.test] == 1)
        confidence = np.maximum(p_up, 1 - p_up)
        for share in TOP_SHARES:
            cut = -np.inf if share >= 1 else np.quantile(p_move_cal, 1 - share)  # threshold from the past only
            for conf in CONFIDENCE:
                mask = (p_move >= cut) & (confidence >= conf)
                idx = independent_positions(np.flatnonzero(mask), h)
                acc = pooled[(share, conf)]
                acc[0] += len(idx)
                acc[1] += int(correct[idx].sum())
                acc[2] += float(np.abs(ret[split.test][idx]).sum())

    print("Direction 4h x volatility filter (+-1% within 4h), 5 walk-forward test periods, independent signals")
    print(f"{'hours kept by filter':<22}{'direction conf.':<17}{'signals':>8}{'hit rate':>10}{'95% lower':>11}{'avg |move|':>12}")
    report = []
    for share in TOP_SHARES:
        for conf in CONFIDENCE:
            n, hits, abs_sum = pooled[(share, conf)]
            hit = hits / n if n else None
            low = wilson_lower_bound(hits, n)
            label = "all hours" if share >= 1 else f"top {share:.0%} P(move)"
            print(f"{label:<22}{'>= ' + format(conf, '.0%'):<17}{n:>8}{'  n/a' if hit is None else format(hit, '10.1%')}"
                  f"{'  n/a' if low is None else format(low, '11.1%')}{'' if not n else format(abs_sum / n, '12.2%')}")
            report.append({"top_share": share, "min_confidence": conf, "independent_signals": n, "hits": hits,
                           "hit_rate": hit, "hit_rate_lower_95": low, "mean_abs_return": abs_sum / n if n else None})

    out = data_dir / "reports" / f"volatility_filter_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"direction": "4h old logic", "volatility": "move_1pct_4h", "rows": report}, indent=2), encoding="utf-8")
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
