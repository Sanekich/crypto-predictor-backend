"""Metrics for return forecasts."""

import math

import numpy as np


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
