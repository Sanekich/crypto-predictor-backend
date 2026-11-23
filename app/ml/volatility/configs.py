"""Configurations of the volatility ("big move") models.

Label: 1 if the price touches +X% or -X% (from the close of the signal candle)
within N hours, 0 otherwise. Thresholds were chosen on BTCUSDT 1h history so that a
big move happens in roughly half of the cases (no trivial "always yes" answer):

    +-1% within 4h    54% of hours overall, 47% since 2021, 37% since 2025
    +-2% within 12h   50% overall, 43% since 2021, 32% since 2025
    +-3% within 24h   49% overall, 42% since 2021, 31% since 2025

The share keeps falling as BTC matures, so the model must read the current regime
from recent volatility instead of relying on the long-run average.
"""

from dataclasses import asdict, dataclass
from typing import Any

from app.features.targets import horizon_to_periods
from app.ml.barrier.configs import FEATURE_SET_LABELS, FEATURE_SETS  # noqa: F401  (same feature sets)

ARTIFACTS_SUBDIR = "volatility"
MODEL_TYPE = "xgboost_volatility"
SIGNAL_THRESHOLD = 0.75


@dataclass(frozen=True)
class VolatilityConfig:
    key: str
    threshold: float  # 0.01 == 1%
    horizon: str
    label: str

    def periods(self, interval: str) -> int:
        return horizon_to_periods(self.horizon, interval)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


VOLATILITY_CONFIGS: dict[str, VolatilityConfig] = {
    c.key: c
    for c in (
        VolatilityConfig("move_1pct_4h", 0.01, "4h", "движение ±1% за 4 часа"),
        VolatilityConfig("move_2pct_12h", 0.02, "12h", "движение ±2% за 12 часов"),
        VolatilityConfig("move_3pct_24h", 0.03, "24h", "движение ±3% за 24 часа"),
    )
}


def variant_key(config_key: str, feature_set: str) -> str:
    return f"{config_key}:{feature_set}"
