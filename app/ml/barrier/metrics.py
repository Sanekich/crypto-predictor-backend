"""Quality of barrier direction probabilities and of high-confidence signals.

Differences from the old-logic metrics:
- accuracy and Brier are computed on resolved rows only (one barrier was hit);
- confident signals are additionally split into hit / miss / no move / ambiguous;
- independent signals are counted with the real resolution time of each signal
  (a signal "occupies" the market until its barrier is hit), not a fixed horizon.
"""

from typing import Any

import numpy as np

from app.ml.evaluation.metrics import SIGNAL_THRESHOLDS, wilson_lower_bound


def independent_signal_positions(positions: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Greedily keep a signal and skip every later signal that starts before its outcome is known."""
    kept: list[int] = []
    busy_until = -1
    for pos, duration in zip(positions, durations, strict=True):
        if pos > busy_until:
            kept.append(int(pos))
            busy_until = int(pos) + int(duration)
    return np.asarray(kept, dtype=int)


def barrier_metrics(
    outcome: np.ndarray,
    p_up: np.ndarray,
    resolve_periods: np.ndarray,
    thresholds: tuple[float, ...] = SIGNAL_THRESHOLDS,
) -> dict[str, Any]:
    outcome = np.asarray(outcome, dtype=object)
    p = np.clip(np.asarray(p_up, dtype="float64"), 1e-6, 1 - 1e-6)
    durations = np.asarray(resolve_periods, dtype=int)
    n = int(outcome.size)
    if n == 0:
        raise ValueError("no rows to evaluate")

    is_up = outcome == "up"
    resolved = is_up | (outcome == "down")
    n_resolved = int(resolved.sum())
    confidence = np.maximum(p, 1 - p)
    predicted_up = p >= 0.5
    correct = predicted_up == is_up

    result: dict[str, Any] = {
        "n": n,
        "n_resolved": n_resolved,
        "resolved_share": n_resolved / n,
        "none_share": float(np.mean(outcome == "none")),
        "both_share": float(np.mean(outcome == "both")),
        "max_confidence": float(confidence.max()),
        "base_rate_up": None,
        "accuracy": None,
        "majority_baseline_accuracy": None,
        "brier": None,
        "brier_base_rate": None,
        "brier_skill": None,
    }
    if n_resolved:
        y = is_up[resolved].astype("float64")
        pr = p[resolved]
        base_rate = float(y.mean())
        brier = float(np.mean((pr - y) ** 2))
        brier_base = float(np.mean((base_rate - y) ** 2))
        result.update(
            base_rate_up=base_rate,
            accuracy=float(correct[resolved].mean()),
            majority_baseline_accuracy=max(base_rate, 1 - base_rate),
            brier=brier,
            brier_base_rate=brier_base,
            brier_skill=(1 - brier / brier_base) if brier_base > 0 else None,
        )

    signals: dict[str, Any] = {}
    for t in thresholds:
        mask = confidence >= t
        positions = np.flatnonzero(mask)
        independent = independent_signal_positions(positions, durations[positions])
        ind_resolved = resolved[independent]
        hits = int((correct[independent] & ind_resolved).sum())
        n_ind_resolved = int(ind_resolved.sum())
        signals[f"{t:.2f}"] = {
            "signals": int(mask.sum()),
            "share_of_time": float(mask.mean()),
            "hit_rate": float(correct[mask & resolved].mean()) if (mask & resolved).any() else None,
            "unresolved_share": float((~resolved[mask]).mean()) if mask.any() else None,
            "independent_signals": int(len(independent)),
            "independent_resolved": n_ind_resolved,
            "independent_hits": hits,
            "independent_hit_rate": hits / n_ind_resolved if n_ind_resolved else None,
            "hit_rate_lower_95": wilson_lower_bound(hits, n_ind_resolved),
            "independent_none": int((outcome[independent] == "none").sum()),
            "independent_both": int((outcome[independent] == "both").sum()),
        }
    result["signals"] = signals
    return result


def pool_signals(fold_metrics: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Sum independent signals over walk-forward test periods: more evidence than one test period."""
    key = f"{threshold:.2f}"
    stats = [m["signals"][key] for m in fold_metrics]
    independent = sum(s["independent_signals"] for s in stats)
    resolved = sum(s["independent_resolved"] for s in stats)
    hits = sum(s["independent_hits"] for s in stats)
    none = sum(s["independent_none"] for s in stats)
    return {
        "threshold": threshold,
        "signals": sum(s["signals"] for s in stats),
        "independent_signals": independent,
        "independent_resolved": resolved,
        "independent_hits": hits,
        "independent_hit_rate": hits / resolved if resolved else None,
        "hit_rate_lower_95": wilson_lower_bound(hits, resolved),
        "independent_none": none,
        "unresolved_share": (independent - resolved) / independent if independent else None,
    }
