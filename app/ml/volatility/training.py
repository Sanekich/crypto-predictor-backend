"""Training and walk-forward verification of the volatility models.

The classifier is the same calibrated XGBoost used for direction (generic binary
model with Platt calibration on a separate period). Each walk-forward fold also
fits a naive baseline - logistic regression on one feature, the last 24 hours'
realized volatility - so the report shows whether the full model adds anything
beyond plain volatility clustering.

Gates (same spirit as the direction models):
- skill: mean Brier skill > 0 and positive on at least 4 of 5 folds;
- approved: additionally, pooled confident signals (move or calm, >= 75%) reach
  at least 77 independent cases with hit rate >= 75% and 95% lower bound >= 70%.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.data.intervals import interval_to_timedelta
from app.ml.evaluation.metrics import SIGNAL_THRESHOLDS
from app.ml.evaluation.splits import chronological_split
from app.ml.evaluation.walk_forward import expanding_folds
from app.ml.models.direction_model import XGBoostDirectionModel
from app.ml.volatility.configs import SIGNAL_THRESHOLD
from app.ml.volatility.dataset import BASELINE_FEATURE, VolatilityDataset
from app.ml.volatility.metrics import pool, volatility_metrics

logger = logging.getLogger(__name__)

GATE = {"min_independent_signals": 77, "min_hit_rate": 0.75, "min_hit_rate_lower_95": 0.70}
SKILL_GATE = {"folds": 5, "min_positive_folds": 4}


def _fit(X, y, train, early_stop, calibration, params) -> XGBoostDirectionModel:
    model = XGBoostDirectionModel(params)
    model.fit(X[train], y[train], X[early_stop], y[early_stop])
    return model.calibrate(X[calibration], y[calibration])


def _baseline_proba(frame, y, train, test) -> np.ndarray:
    x = np.log(frame[BASELINE_FEATURE].to_numpy(dtype="float64") + 1e-9).reshape(-1, 1)
    return LogisticRegression().fit(x[train], y[train]).predict_proba(x[test])[:, 1]


def _gate(pooled: dict[str, Any]) -> dict[str, Any]:
    n, hit, low = pooled["independent_signals"], pooled["independent_hit_rate"], pooled["hit_rate_lower_95"]
    reasons = []
    if n < GATE["min_independent_signals"]:
        reasons.append(f"only {n} independent signals >= {SIGNAL_THRESHOLD:.0%} (need {GATE['min_independent_signals']})")
    if hit is not None and hit < GATE["min_hit_rate"]:
        reasons.append(f"hit rate {hit:.1%} below {GATE['min_hit_rate']:.0%}")
    if low is not None and low < GATE["min_hit_rate_lower_95"]:
        reasons.append(f"95% lower bound {low:.1%} below {GATE['min_hit_rate_lower_95']:.0%}")
    return {"passed": not reasons, "reasons": reasons}


def walk_forward(dataset: VolatilityDataset, folds: int = SKILL_GATE["folds"], params: dict[str, Any] | None = None) -> dict[str, Any]:
    frame = dataset.frame.sort_values("timestamp").reset_index(drop=True)
    X = frame[dataset.feature_columns].to_numpy(dtype="float64")
    y = frame["label"].to_numpy(dtype="float64")
    ts = frame["timestamp"]

    results: list[dict[str, Any]] = []
    for number, split in enumerate(expanding_folds(ts, dataset.periods, folds), start=1):
        model = _fit(X, y, split.train, split.early_stopping, split.calibration, params)
        metrics = volatility_metrics(y[split.test], model.predict_proba_up(X[split.test]), dataset.periods)
        baseline = volatility_metrics(y[split.test], _baseline_proba(frame, y, split.train, split.test), dataset.periods)
        metrics.update(
            fold=number,
            rows=int(len(split.test)),
            test_start=ts.iloc[split.test[0]].isoformat(),
            test_end=ts.iloc[split.test[-1]].isoformat(),
            baseline_brier_skill=baseline["brier_skill"],
            baseline_auc=baseline["auc"],
        )
        results.append(metrics)

    skills = [m["brier_skill"] for m in results if m["brier_skill"] is not None]
    baseline_skills = [m["baseline_brier_skill"] for m in results if m["baseline_brier_skill"] is not None]
    positive = sum(s > 0 for s in skills)
    mean_skill = float(np.mean(skills)) if skills else None
    required = min(SKILL_GATE["min_positive_folds"], len(skills))
    pooled = {
        kind: {f"{t:.2f}": pool(results, kind, t) for t in SIGNAL_THRESHOLDS} if results else {}
        for kind in ("move_signals", "calm_signals")
    }
    best_kind = max(
        ("move_signals", "calm_signals"),
        key=lambda k: (pooled[k].get(f"{SIGNAL_THRESHOLD:.2f}") or {}).get("hit_rate_lower_95") or 0,
    )
    empty = {"independent_signals": 0, "independent_hit_rate": None, "hit_rate_lower_95": None}
    best = pooled[best_kind].get(f"{SIGNAL_THRESHOLD:.2f}", empty) if results else empty
    return {
        "folds": len(results),
        "by_fold": results,
        "brier_skill_by_fold": skills,
        "brier_skill_mean": mean_skill,
        "baseline_brier_skill_by_fold": baseline_skills,
        "baseline_brier_skill_mean": float(np.mean(baseline_skills)) if baseline_skills else None,
        "auc_by_fold": [m["auc"] for m in results],
        "positive_folds": positive,
        "required_positive_folds": required,
        "has_skill": mean_skill is not None and mean_skill > 0 and positive >= required,
        "beats_baseline": bool(skills and baseline_skills and np.mean(skills) > np.mean(baseline_skills)),
        "pooled": pooled,
        "best_signal_kind": best_kind,
        "gate": {**_gate(best), "signal_kind": best_kind, "criteria": GATE},
    }


@dataclass
class VolatilityTrainingResult:
    model: XGBoostDirectionModel
    split_info: dict[str, Any]
    metrics: dict[str, Any]
    walk_forward: dict[str, Any]

    @property
    def status(self) -> str:
        wf = self.walk_forward
        return "approved" if wf["has_skill"] and wf["gate"]["passed"] else "shadow"


def train_volatility_model(
    dataset: VolatilityDataset,
    params: dict[str, Any] | None = None,
    folds: int = SKILL_GATE["folds"],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.15,
) -> VolatilityTrainingResult:
    frame = dataset.frame.sort_values("timestamp").reset_index(drop=True)
    gap = interval_to_timedelta(dataset.interval) * dataset.periods
    split = chronological_split(frame["timestamp"], gap, validation_fraction, test_fraction)
    half = len(split.validation) // 2
    early_stop = split.validation[:half]
    calibration = split.validation[half + dataset.periods :]
    if len(early_stop) == 0 or len(calibration) == 0:
        raise ValueError("not enough validation data for early stopping and calibration")

    X = frame[dataset.feature_columns].to_numpy(dtype="float64")
    y = frame["label"].to_numpy(dtype="float64")
    model = _fit(X, y, split.train, early_stop, calibration, params)
    test = volatility_metrics(y[split.test], model.predict_proba_up(X[split.test]), dataset.periods)
    test["baseline"] = volatility_metrics(y[split.test], _baseline_proba(frame, y, split.train, split.test), dataset.periods)

    ts = frame["timestamp"]
    split_info = split.describe(ts)
    split_info["early_stopping"] = {"rows": len(early_stop), "start": ts.iloc[early_stop[0]].isoformat(), "end": ts.iloc[early_stop[-1]].isoformat()}
    split_info["calibration"] = {"rows": len(calibration), "start": ts.iloc[calibration[0]].isoformat(), "end": ts.iloc[calibration[-1]].isoformat()}

    verification = walk_forward(dataset, folds, params)
    logger.info("Volatility model %s/%s: test skill %s, walk-forward skill %s",
                dataset.config.key, dataset.feature_set, test["brier_skill"], verification["brier_skill_mean"])
    return VolatilityTrainingResult(model, split_info, {"test": test}, verification)
