"""Event study: what happens after extreme euphoria or panic?

An event starts when the index crosses into a zone (>= +threshold: euphoria,
<= -threshold: panic). After an event the next one of the same kind is counted only
once the horizon has passed, so every event is independent evidence.

Hypothesis "reversal": after euphoria the price is lower `horizon` hours later,
after panic it is higher. For each zone the hit rate is compared with the
unconditional base rate of that direction in the same years (BTC rises slightly
more often than it falls), and the results are split by year to see whether the
effect is stable or comes from one market phase.

Thresholds are fixed in z-score units and every component is a causal rolling
z-score, so no future information decides what counts as an event.
"""

from typing import Any

import numpy as np
import pandas as pd

from app.ml.evaluation.metrics import wilson_lower_bound

THRESHOLDS = (1.0, 1.5, 2.0)
HORIZONS = {"24h": 24, "72h": 72, "168h": 168}
GATE = {"min_independent_signals": 77, "min_hit_rate": 0.75, "min_hit_rate_lower_95": 0.70}


def event_positions(in_zone: np.ndarray, spacing: int) -> np.ndarray:
    """First hour of each stay in the zone, at least `spacing` hours after the previous event."""
    events: list[int] = []
    previous = False
    for i, flag in enumerate(in_zone):
        if flag and not previous and (not events or i - events[-1] >= spacing):
            events.append(i)
        previous = bool(flag)
    return np.asarray(events, dtype=int)


def _stats(hit: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    n, hits = int(hit.size), int(hit.sum())
    return {
        "events": n,
        "hits": hits,
        "hit_rate": hits / n if n else None,
        "hit_rate_lower_95": wilson_lower_bound(hits, n),
        "mean_return": float(returns.mean()) if n else None,
        "median_return": float(np.median(returns)) if n else None,
    }


def run_study(frame: pd.DataFrame) -> dict[str, Any]:
    """frame: build_euphoria_frame output. Returns per zone / threshold / horizon statistics."""
    frame = frame.reset_index(drop=True)
    close = frame["close"].to_numpy(dtype="float64")
    score = frame["euphoria"].to_numpy(dtype="float64")
    years = frame["timestamp"].dt.year.to_numpy()
    valid = ~np.isnan(score)
    results: list[dict[str, Any]] = []

    for horizon_key, h in HORIZONS.items():
        future = np.full(close.size, np.nan)
        future[:-h] = close[h:] / close[:-h] - 1
        known = valid & ~np.isnan(future)
        up_share = float((future[known] > 0).mean()) if known.any() else None
        for threshold in THRESHOLDS:
            for zone, sign in (("euphoria", 1), ("panic", -1)):
                in_zone = known & (sign * score >= threshold)
                events = event_positions(in_zone, h)
                ret = future[events]
                expected_down = zone == "euphoria"
                hit = (ret < 0) if expected_down else (ret > 0)
                base = None if up_share is None else (1 - up_share if expected_down else up_share)
                by_year = {}
                for year in np.unique(years[events]) if events.size else []:
                    mask = years[events] == year
                    by_year[str(int(year))] = _stats(hit[mask], ret[mask])
                stats = _stats(hit, ret)
                results.append({
                    "zone": zone,
                    "expected": "down" if expected_down else "up",
                    "threshold": threshold,
                    "horizon": horizon_key,
                    **stats,
                    "base_rate": base,
                    "edge_vs_base": None if stats["hit_rate"] is None or base is None else stats["hit_rate"] - base,
                    "positive_years": sum(1 for s in by_year.values() if s["hit_rate"] is not None and s["hit_rate"] > (base or 0.5)),
                    "years": len(by_year),
                    "by_year": by_year,
                    "passes_gate": bool(
                        stats["events"] >= GATE["min_independent_signals"]
                        and (stats["hit_rate"] or 0) >= GATE["min_hit_rate"]
                        and (stats["hit_rate_lower_95"] or 0) >= GATE["min_hit_rate_lower_95"]
                    ),
                })
    first = frame.loc[valid, "timestamp"]
    return {
        "period": {"start": first.iloc[0].isoformat() if len(first) else None, "end": first.iloc[-1].isoformat() if len(first) else None},
        "hours_with_index": int(valid.sum()),
        "gate": GATE,
        "results": results,
    }


def best_results(study: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """Most convincing rows: by 95% lower bound of the hit rate, at least 10 events."""
    rows = [r for r in study["results"] if r["events"] >= 10 and r["hit_rate_lower_95"] is not None]
    return sorted(rows, key=lambda r: r["hit_rate_lower_95"], reverse=True)[:limit]
