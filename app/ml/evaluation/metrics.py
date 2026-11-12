"""Metrics for return forecasts."""

import math
from typing import Any

import numpy as np


SIGNAL_THRESHOLDS = (0.6, 0.65, 0.7, 0.75, 0.8)


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float | None:
    """Lower end of the 95% confidence interval for a hit rate (Wilson score)."""
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denominator


def independent_positions(positions: np.ndarray, spacing: int) -> np.ndarray:
    """Greedily keep positions at least `spacing` rows apart.

    Signals on overlapping horizons (e.g. every hour for a 4-hour horizon)
    describe the same price move and must not be counted as separate evidence.
    """
    kept: list[int] = []
    for pos in positions:
        if not kept or pos - kept[-1] >= spacing:
            kept.append(int(pos))
    return np.asarray(kept, dtype=int)


def independent_sample_indices(indices: np.ndarray, spacing: int) -> np.ndarray:
    """Keep a regular, non-overlapping subset of row indices for calibration/evaluation."""
    values = np.asarray(indices, dtype=int)
    if spacing <= 1 or values.size <= 1:
        return values
    return values[::spacing]


def direction_metrics(
    y_up: np.ndarray,
    p_up: np.ndarray,
    horizon_periods: int = 1,
    thresholds: tuple[float, ...] = SIGNAL_THRESHOLDS,
) -> dict[str, Any]:
    """Quality of direction probabilities and of high-confidence signals.

    A signal at threshold t is any row with max(p, 1 - p) >= t, predicting
    "up" if p >= 0.5. For each threshold: how often signals occur, how often
    they were right, and the 95% lower bound of that hit rate computed on
    independent (non-overlapping) signals only.
    """
    y = np.asarray(y_up, dtype="float64")
    p = np.clip(np.asarray(p_up, dtype="float64"), 1e-6, 1 - 1e-6)
    base_rate = float(np.mean(y))
    predicted_up = p >= 0.5
    correct = predicted_up == (y == 1)
    confidence = np.maximum(p, 1 - p)

    signals: dict[str, Any] = {}
    for t in thresholds:
        mask = confidence >= t
        positions = np.flatnonzero(mask)
        independent = independent_positions(positions, horizon_periods)
        hits = int(correct[independent].sum())
        signals[f"{t:.2f}"] = {
            "signals": int(mask.sum()),
            "share_of_time": float(mask.mean()),
            "hit_rate": float(correct[mask].mean()) if mask.any() else None,
            "independent_signals": int(len(independent)),
            "independent_hit_rate": hits / len(independent) if len(independent) else None,
            "hit_rate_lower_95": wilson_lower_bound(hits, len(independent)),
        }

    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    base = np.clip(base_rate, 1e-6, 1 - 1e-6)
    return {
        "n": int(y.size),
        "base_rate_up": base_rate,
        "accuracy": float(np.mean(correct)),
        "always_up_accuracy": max(base_rate, 1 - base_rate),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_base_rate": float(np.mean((base_rate - y) ** 2)),
        "log_loss": log_loss,
        "log_loss_base_rate": float(-(base_rate * np.log(base) + (1 - base_rate) * np.log(1 - base))),
        "max_confidence": float(confidence.max()),
        "signals": signals,
    }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Standard loss for quantile forecasts (lower is better)."""
    diff = np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_metrics(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[str, float | int]:
    """Quality of a range forecast built from the outer quantiles.

    - coverage: share of outcomes inside [lowest, highest] quantile; should be
      close to nominal_coverage (e.g. 0.8 for the 0.1..0.9 range);
    - mean_width / mean_width_bps: average range width (narrower is better
      at the same coverage);
    - pinball_mean: average pinball loss over all quantiles;
    - width_correlation: does the predicted width follow actual |return|?
      Positive means the model knows when the market will be more volatile.
    """
    y_true = np.asarray(y_true, dtype="float64")
    preds = np.asarray(quantile_preds, dtype="float64")
    lower, upper = preds[:, 0], preds[:, -1]
    width = upper - lower
    covered = (y_true >= lower) & (y_true <= upper)

    result: dict[str, float | int] = {
        "n": int(y_true.size),
        "nominal_coverage": round(quantiles[-1] - quantiles[0], 6),
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "mean_width_bps": float(np.mean(width) * 10_000),
        "pinball_mean": float(np.mean([pinball_loss(y_true, preds[:, i], q) for i, q in enumerate(quantiles)])),
    }
    for i, q in enumerate(quantiles):
        result[f"below_q{q:g}"] = float(np.mean(y_true < preds[:, i]))
    if np.std(width) > 1e-12:  # a constant range (baseline) has no correlation
        result["width_correlation"] = float(np.corrcoef(width, np.abs(y_true))[0, 1])
    else:
        result["width_correlation"] = None  # type: ignore[assignment]
    return result


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    """Error metrics for predicted returns (fractions, e.g. 0.01 == 1%).

    - mae / rmse / mse: size of the error;
    - mae_bps: MAE in basis points (1 bp = 0.01%), easier to read;
    - direction_accuracy: share of rows where "return > 0" was predicted correctly;
    - correlation: Pearson correlation between forecast and outcome
      (None if a forecast is constant, e.g. the zero baseline).
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and of equal shape")

    error = y_pred - y_true
    mse = float(np.mean(error**2))
    correlation = None
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(error))),
        "mae_bps": float(np.mean(np.abs(error)) * 10_000),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "direction_accuracy": float(np.mean((y_pred > 0) == (y_true > 0))),
        "correlation": correlation,
    }
