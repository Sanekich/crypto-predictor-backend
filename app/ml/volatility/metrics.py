"""Quality of "big move" probabilities and of confident move / calm signals."""

from typing import Any

import numpy as np

from app.ml.evaluation.metrics import SIGNAL_THRESHOLDS, independent_positions, wilson_lower_bound


def _signal_stats(hit: np.ndarray, mask: np.ndarray, spacing: int) -> dict[str, Any]:
    positions = np.flatnonzero(mask)
    independent = independent_positions(positions, spacing)
    hits = int(hit[independent].sum())
    n = int(len(independent))
    return {
        "signals": int(mask.sum()),
        "share_of_time": float(mask.mean()),
        "hit_rate": float(hit[mask].mean()) if mask.any() else None,
        "independent_signals": n,
        "independent_hits": hits,
        "independent_hit_rate": hits / n if n else None,
        "hit_rate_lower_95": wilson_lower_bound(hits, n),
    }


def volatility_metrics(
    y_move: np.ndarray,
    p_move: np.ndarray,
    spacing: int,
    thresholds: tuple[float, ...] = SIGNAL_THRESHOLDS,
) -> dict[str, Any]:
    """Brier skill vs the period's own base rate, plus confident signals of both kinds.

    "move" signal at t: P(big move) >= t, correct if the move happened.
    "calm" signal at t: P(big move) <= 1 - t, correct if it did not.
    Independent signals are at least `spacing` candles (the horizon) apart.
    """
    y = np.asarray(y_move, dtype="float64")
    p = np.clip(np.asarray(p_move, dtype="float64"), 1e-6, 1 - 1e-6)
    if y.size == 0:
        raise ValueError("no rows to evaluate")
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    order = np.argsort(p)
    # AUC via ranks: probability that a random "move" row gets a higher score than a "calm" row
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else None

    move_hit = y == 1
    calm_hit = y == 0
    return {
        "n": int(y.size),
        "move_share": base,
        "mean_p_move": float(p.mean()),
        "accuracy": float(np.mean((p >= 0.5) == move_hit)),
        "majority_baseline_accuracy": max(base, 1 - base),
        "brier": brier,
        "brier_base_rate": brier_base,
        "brier_skill": (1 - brier / brier_base) if brier_base > 0 else None,
        "auc": auc,
        "max_p_move": float(p.max()),
        "min_p_move": float(p.min()),
        "move_signals": {f"{t:.2f}": _signal_stats(move_hit, p >= t, spacing) for t in thresholds},
        "calm_signals": {f"{t:.2f}": _signal_stats(calm_hit, p <= 1 - t, spacing) for t in thresholds},
    }


def pool(fold_metrics: list[dict[str, Any]], kind: str, threshold: float) -> dict[str, Any]:
    """Sum independent signals over walk-forward test periods."""
    key = f"{threshold:.2f}"
    stats = [m[kind][key] for m in fold_metrics]
    n = sum(s["independent_signals"] for s in stats)
    hits = sum(s["independent_hits"] for s in stats)
    return {
        "threshold": threshold,
        "signals": sum(s["signals"] for s in stats),
        "independent_signals": n,
        "independent_hits": hits,
        "independent_hit_rate": hits / n if n else None,
        "hit_rate_lower_95": wilson_lower_bound(hits, n),
    }
