"""Configurations of the barrier ("significant move") direction models.

Old logic: "will the price be higher after N hours?" Most such moves are noise of a
fraction of a percent, which caps the model's confidence at about 60-65%.

New logic: "which comes first, a rise of X% or a fall of X%?", waiting at most
`max_horizon`. Thresholds and waiting limits were chosen on BTCUSDT 1h history so
that 90-96% of cases resolve (one of the barriers is reached):

    +-1% within 24h   91% resolved, median 4 hours to a hit
    +-2% within 72h   94% resolved, median 12 hours
    +-3% within 168h  96% resolved, median 24 hours

Each configuration is trained on two feature sets: candles only (history from 2017)
and all data (candles + funding + basis + open interest + DVOL, history from 2021-03).
"""

from dataclasses import asdict, dataclass
from typing import Any

from app.features.derivatives import GROUPS as DERIVATIVE_GROUPS
from app.features.targets import horizon_to_periods

# artifacts/<ARTIFACTS_SUBDIR>/models/<model_id>: never mixed with the old logic in artifacts/models
ARTIFACTS_SUBDIR = "barrier"
MODEL_TYPE = "xgboost_barrier_direction"
SIGNAL_THRESHOLD = 0.75


@dataclass(frozen=True)
class BarrierConfig:
    key: str
    threshold: float  # 0.01 == 1%
    max_horizon: str  # waiting limit, e.g. "24h"
    label: str

    def max_periods(self, interval: str) -> int:
        return horizon_to_periods(self.max_horizon, interval)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


BARRIER_CONFIGS: dict[str, BarrierConfig] = {
    c.key: c
    for c in (
        BarrierConfig("move_1pct_24h", 0.01, "24h", "±1% в течение 24 часов"),
        BarrierConfig("move_2pct_72h", 0.02, "72h", "±2% в течение 3 дней"),
        BarrierConfig("move_3pct_168h", 0.03, "168h", "±3% в течение 7 дней"),
    )
}

# feature set key -> derivatives groups added to the candle features
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "candles": (),
    "all_data": tuple(DERIVATIVE_GROUPS),
}
FEATURE_SET_LABELS: dict[str, str] = {
    "candles": "только свечи",
    "all_data": "свечи + funding + basis + open interest + DVOL",
}


def variant_key(config_key: str, feature_set: str) -> str:
    """Models with the same variant are versions of the same experiment."""
    return f"{config_key}:{feature_set}"
