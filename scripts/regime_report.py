"""Report direction quality by causal market regime.

Examples:
    python scripts/regime_report.py --horizon 24h
    python scripts/regime_report.py --horizon 7d

The report does not retrain, save models, or change API behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.data.intervals import interval_to_timedelta  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402
from app.features.feature_pipeline import FeatureConfig, build_dataset  # noqa: E402
from app.ml.evaluation.regimes import REGIMES, label_regimes, regime_features  # noqa: E402
from app.ml.evaluation.splits import chronological_split  # noqa: E402
from app.ml.models.direction_model import DirectionModel  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry  # noqa: E402
from app.services.features_service import check_compatibility, latest_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizon", default="24h")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    artifacts_dir = settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    ohlcv = MarketDataStorage(data_dir).load_processed(settings.market_data_provider, args.symbol, args.interval)
    if ohlcv is None:
        print("No processed data. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    dataset = build_dataset(ohlcv, args.interval, args.horizon)
    frame = dataset.frame
    gap = interval_to_timedelta(args.interval) * dataset.horizon_periods
    split = chronological_split(frame["timestamp"], gap)
    registry = ModelRegistry(artifacts_dir)
    meta = registry.latest("xgboost_direction", args.symbol, args.interval, args.horizon)
    if meta is None:
        print("No saved direction model for this horizon.", file=sys.stderr)
        return 1
    check_compatibility(meta, FeatureConfig())
    model, _ = registry.load(meta["model_id"])
    if not isinstance(model, DirectionModel):
        print("Saved model is not a direction model.", file=sys.stderr)
        return 1

    features = regime_features(frame["close"])
    calibration_mask = frame.index.isin(split.train)
    regimes = label_regimes(features, calibration_mask)
    x_test = frame[meta["feature_columns"]].to_numpy(dtype="float64")[split.test]
    y_test = frame[dataset.target_direction_column].to_numpy(dtype="float64")[split.test]
    p_up = model.predict_proba_up(x_test)
    predicted_up = p_up >= 0.5
    correct = predicted_up == (y_test == 1)
    confidence = np.maximum(p_up, 1 - p_up)
    regime_test = regimes.iloc[split.test]

    print(f"Regime report: {args.symbol} {args.interval} {args.horizon}")
    print(f"model: {meta['model_id']} ({meta.get('status', 'n/a')})")
    print(f"{'regime':<18} {'rows':>7} {'accuracy':>10} {'baseline':>10} {'mean conf':>10}")
    for regime in REGIMES:
        mask = regime_test.to_numpy() == regime
        if not mask.any():
            print(f"{regime:<18} {0:>7} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
            continue
        base = max(float(np.mean(y_test[mask])), 1 - float(np.mean(y_test[mask])))
        print(f"{regime:<18} {int(mask.sum()):>7} {float(correct[mask].mean()):>9.1%} {base:>9.1%} {float(confidence[mask].mean()):>9.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
