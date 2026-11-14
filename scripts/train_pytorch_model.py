"""Train the independent PyTorch GRU challenger on 168-hour market sequences."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.data.derivatives.store import DerivativesStore  # noqa: E402
from app.data.storage import MarketDataStorage  # noqa: E402
from app.features.feature_pipeline import FeatureConfig, build_dataset  # noqa: E402
from app.ml.models.pytorch_gru_model import PyTorchGRUDirectionModel  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry, make_model_id  # noqa: E402


def sequences(dataset, length: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = dataset.frame
    cols = dataset.feature_columns
    x, y, stamps = [], [], []
    values = frame[cols].to_numpy(dtype=np.float32)
    target = frame[dataset.target_direction_column].to_numpy(dtype=np.float32)
    ts = frame["timestamp"].reset_index(drop=True)
    for end in range(length - 1, len(frame)):
        window = ts.iloc[end - length + 1:end + 1]
        if (window.diff().dropna() != np.timedelta64(1, "h")).any():
            continue
        x.append(values[end - length + 1:end + 1])
        y.append(target[end])
        stamps.append(ts.iloc[end].isoformat())
    if not x:
        raise RuntimeError("No contiguous 168-hour sequences are available")
    return np.stack(x), np.asarray(y), stamps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizon", choices=("24h", "7d"), default="24h")
    parser.add_argument("--sequence-length", type=int, default=168)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--shadow-only", action="store_true", help="required for saved artifacts")
    args = parser.parse_args(argv)
    if not args.no_save and not args.shadow_only:
        print("Refusing to save a PyTorch model without --shadow-only", file=sys.stderr)
        return 2

    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    ohlcv_path = data_dir / "processed" / settings.market_data_provider / f"{args.symbol}_{args.interval}.parquet"
    if not ohlcv_path.exists():
        print(f"No processed OHLCV at {ohlcv_path}", file=sys.stderr)
        return 1
    try:
        ohlcv = MarketDataStorage(data_dir).load_processed(settings.market_data_provider, args.symbol, args.interval)
        if ohlcv is None:
            raise FileNotFoundError(f"No processed OHLCV at {ohlcv_path}")
        derivatives = DerivativesStore(data_dir).load_all(args.symbol)
        config = FeatureConfig(derivatives=("funding", "basis", "open_interest", "dvol"))
        dataset = build_dataset(ohlcv, args.interval, args.horizon, config, derivatives)
        X, y, stamps = sequences(dataset, args.sequence_length)
    except Exception as exc:
        print(f"Cannot prepare PyTorch full-derivatives dataset: {exc}", file=sys.stderr)
        return 1

    n = len(X)
    train_end, val_end = int(n * 0.60), int(n * 0.75)
    if train_end < 100 or n - val_end < 100:
        print(f"Not enough sequences for chronological split: {n}", file=sys.stderr)
        return 1
    model = PyTorchGRUDirectionModel(input_size=X.shape[-1])
    model.fit(X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], epochs=args.epochs)
    p = model.predict_proba_up(X[val_end:])
    pred = (p >= 0.5).astype(int)
    accuracy = accuracy_score(y[val_end:], pred)
    baseline = max(float(y[val_end:].mean()), 1.0 - float(y[val_end:].mean()))
    brier = brier_score_loss(y[val_end:], p)
    print(f"PyTorch GRU direction: test accuracy {accuracy:.3f} (majority {baseline:.3f}), status shadow")
    print(f"Sequences           : {n} (length {args.sequence_length}h, {X.shape[-1]} features)")
    print(f"train               : {train_end} rows  {stamps[0]} .. {stamps[train_end - 1]}")
    print(f"validation          : {val_end - train_end} rows  {stamps[train_end]} .. {stamps[val_end - 1]}")
    print(f"test                : {n - val_end} rows  {stamps[val_end]} .. {stamps[-1]}")
    print("Feature set         : full derivatives (OHLCV + funding + basis + open interest + DVOL)")
    print(f"Test accuracy       : {accuracy:.1%}")
    print(f"Majority baseline   : {baseline:.1%}")
    print(f"Brier score         : {brier:.4f}")
    print(f"Max confidence      : {max(float(p.max()), float(1 - p.min())):.1%}")
    print("Status              : SHADOW")

    if not args.no_save:
        registry = ModelRegistry(settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir)
        model_id = make_model_id("pytorch-gru-direction-full-derivatives", args.symbol, args.interval, args.horizon)
        registry.save(model, model_id, {
            "symbol": args.symbol.upper(), "interval": args.interval, "horizon": args.horizon,
            "status": "shadow", "trained_at": stamps[-1], "feature_set": "pytorch_full_derivatives",
            "feature_pipeline_version": "1.0", "feature_columns": dataset.feature_columns,
            "feature_config": config.to_dict(), "sequence_length": args.sequence_length,
            "rows": n, "test_accuracy": float(accuracy), "majority_baseline": float(baseline), "brier_score": float(brier),
            "model_params": model.params(), "training_info": model.training_info(),
        }, model_type="pytorch_gru_direction")
        print(f"Saved model        : {model_id}")
    else:
        print("Saved model        : no-save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
