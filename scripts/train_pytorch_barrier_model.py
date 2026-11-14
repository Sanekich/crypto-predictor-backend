"""Train an isolated PyTorch GRU on the new ±1% within 24h barrier target."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.versioning import environment_info, git_revision  # noqa: E402
from app.ml.barrier.configs import BARRIER_CONFIGS, FEATURE_SET_LABELS  # noqa: E402
from app.ml.barrier.dataset import build_barrier_dataset, load_market_data  # noqa: E402
from app.ml.barrier.metrics import barrier_metrics  # noqa: E402
from app.ml.models.pytorch_gru_model import PyTorchGRUDirectionModel  # noqa: E402
from app.ml.registry.model_registry import ModelRegistry, make_model_id  # noqa: E402


def make_sequences(dataset, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = dataset.frame.sort_values("timestamp").reset_index(drop=True)
    cols = dataset.feature_columns
    values = frame[cols].to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy(dtype=np.float32)
    ts = frame["timestamp"].reset_index(drop=True)
    X, y, ends = [], [], []
    for end in range(length - 1, len(frame)):
        window = ts.iloc[end - length + 1:end + 1]
        if (window.diff().dropna() != np.timedelta64(1, "h")).any():
            continue
        X.append(values[end - length + 1:end + 1])
        y.append(labels[end])
        ends.append(end)
    if not X:
        raise RuntimeError("No contiguous 168-hour barrier sequences are available")
    return np.stack(X), np.asarray(y), np.asarray(ends, dtype=int)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--config", default="move_1pct_24h", choices=("move_1pct_24h",))
    parser.add_argument("--feature-set", default="all_data", choices=("all_data",))
    parser.add_argument("--sequence-length", type=int, default=168)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--shadow-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_save and not args.shadow_only:
        print("Refusing to save a PyTorch barrier model without --shadow-only", file=sys.stderr)
        return 2

    settings = get_settings()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    artifacts_dir = settings.model_artifacts_dir if settings.model_artifacts_dir.is_absolute() else ROOT / settings.model_artifacts_dir
    config = BARRIER_CONFIGS[args.config]
    try:
        ohlcv, derivatives = load_market_data(data_dir, settings.market_data_provider, args.symbol.upper(), args.interval, args.feature_set)
        dataset = build_barrier_dataset(ohlcv, args.interval, config, args.feature_set, derivatives)
        X, labels, ends = make_sequences(dataset, args.sequence_length)
    except Exception as exc:
        print(f"Cannot prepare PyTorch barrier dataset: {exc}", file=sys.stderr)
        return 1

    known = ~np.isnan(labels)
    n = len(X)
    train_end, val_end = int(n * 0.60), int(n * 0.75)
    train_known = np.arange(train_end)[known[:train_end]]
    val_known = np.arange(train_end, val_end)[known[train_end:val_end]]
    if len(train_known) < 200 or len(val_known) < 100:
        print(f"Not enough resolved rows for training: train={len(train_known)} validation={len(val_known)}", file=sys.stderr)
        return 1

    model = PyTorchGRUDirectionModel(input_size=X.shape[-1])
    model.fit(X[train_known], labels[train_known], X[val_known], labels[val_known], epochs=args.epochs)
    p = model.predict_proba_up(X[val_end:])
    # Reconstruct outcomes for the test endpoints; barrier frame ordering is preserved.
    outcomes = dataset.frame.sort_values("timestamp").reset_index(drop=True)["outcome"].to_numpy(dtype=object)[ends[val_end:]]
    durations = dataset.frame.sort_values("timestamp").reset_index(drop=True)["resolve_periods"].to_numpy(dtype=int)[ends[val_end:]]
    metrics = barrier_metrics(outcomes, p, durations)
    key = "0.75"
    sig = metrics["signals"][key]
    print(f"PyTorch GRU barrier: ±{config.threshold:.0%} within {config.max_horizon}, status shadow")
    print(f"Sequences           : {n} (length {args.sequence_length}h, {X.shape[-1]} features)")
    print(f"Train resolved      : {len(train_known)}")
    print(f"Validation resolved : {len(val_known)}")
    print(f"Test rows           : {len(p)} (resolved {metrics['n_resolved']} / {metrics['n']})")
    # Keep CLI output ASCII-safe: Windows PowerShell may use cp1252 and cannot
    # encode the Russian human-readable feature-set label.
    print(f"Feature set         : {args.feature_set}")
    print(f"Test accuracy       : {metrics['accuracy']:.1%} (majority {metrics['majority_baseline_accuracy']:.1%})" if metrics['accuracy'] is not None else "Test accuracy       : n/a")
    print(f"Brier skill         : {metrics['brier_skill']:+.4f}" if metrics['brier_skill'] is not None else "Brier skill         : n/a")
    print(f"Signals >= 75%%      : {sig['independent_resolved']} resolved, hit {sig['independent_hit_rate'] if sig['independent_hit_rate'] is not None else 'n/a'}")
    print("Status              : SHADOW")

    if not args.no_save:
        model_id = make_model_id("pytorch-gru-barrier-direction", args.symbol, args.interval, "24h")
        trained_at = datetime.now(timezone.utc).isoformat()
        registry = ModelRegistry(artifacts_dir)
        registry.save(model, model_id, {
            "trained_at": trained_at, "provider": settings.market_data_provider, "symbol": args.symbol.upper(),
            "interval": args.interval, "logic": "barrier", "horizon": config.max_horizon,
            "barrier": config.describe(), "max_periods": dataset.max_periods, "feature_set": args.feature_set,
            "feature_set_label": FEATURE_SET_LABELS[args.feature_set], "variant": f"{config.key}:pytorch_gru",
            "status": "shadow", "feature_columns": dataset.feature_columns, "feature_pipeline_version": "1.0",
            "feature_config": dataset.feature_config.to_dict(), "sequence_length": args.sequence_length,
            "dataset": dataset.describe(), "metrics": {"test": metrics}, "test": metrics,
            "training": model.training_info(), "params": model.params(), "code_version": git_revision(),
            "environment": environment_info(),
        }, model_type="pytorch_gru_barrier_direction")
        print(f"Saved model        : {model_id}")
    else:
        print("Saved model        : no-save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
