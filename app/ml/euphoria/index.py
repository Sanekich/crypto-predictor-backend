"""Hourly euphoria / panic index built from five causal components.

Each component is turned into a rolling z-score over the previous 90 days, so the
index compares the market with its own recent past (BTC at 100k$ in 2025 is not
"euphoric" just because the price is high). Positive = euphoria, negative = panic.

| Component    | Meaning when high                         | Source                       |
|--------------|-------------------------------------------|------------------------------|
| funding      | longs pay shorts: crowd is long           | Binance funding (24 h mean)  |
| oi_growth    | leverage is being added fast              | open interest, 7-day change  |
| crowd_long   | retail accounts are long                  | global long/short ratio      |
| price_stretch| price far above its 30-day average        | candles                      |
| fear_greed   | sentiment index says "greed"              | alternative.me, daily        |

Every input is attached by publication time (as-of joins), so a value is used only
after it was public. The index is NaN until all components are available.
"""

import numpy as np
import pandas as pd

from app.data.processing.resampling import reindex_to_grid
from app.features.derivatives import _hourly_metrics, asof_values

COMPONENTS = ("funding", "oi_growth", "crowd_long", "price_stretch", "fear_greed")
COMPONENT_LABELS = {
    "funding": "Funding (лонги платят шортам)",
    "oi_growth": "Рост открытого интереса за 7 дней",
    "crowd_long": "Доля лонгов у розничных трейдеров",
    "price_stretch": "Отрыв цены от средней за 30 дней",
    "fear_greed": "Индекс страха и жадности",
}
REQUIRED_DATASETS = ("funding", "metrics", "fear_greed")
Z_WINDOW = 24 * 90
Z_MIN_PERIODS = 24 * 30


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    std = series.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std()
    return (series - mean) / std.where(std > 0)


def build_euphoria_frame(ohlcv: pd.DataFrame, data: dict[str, pd.DataFrame], interval: str = "1h") -> pd.DataFrame:
    """timestamp, close, raw components, their z-scores (z_*) and the composite `euphoria`."""
    missing = [name for name in REQUIRED_DATASETS if name not in data]
    if missing:
        raise ValueError(f"euphoria index needs datasets: {', '.join(missing)}")
    if interval != "1h":
        raise ValueError("the euphoria index is defined on 1h candles")

    grid = reindex_to_grid(ohlcv, interval)[["timestamp", "close"]].copy()
    close_times = grid["timestamp"] + pd.Timedelta(hours=1)
    close = grid["close"]

    funding = data["funding"].sort_values("available_at").copy()
    funding["funding_mean_3"] = funding["funding_rate"].rolling(3, min_periods=3).mean()
    metrics = _hourly_metrics(data["metrics"])
    oi = asof_values(close_times, metrics, ["open_interest_value", "global_account_ratio"], pd.Timedelta(minutes=30))
    log_oi = np.log(oi["open_interest_value"].where(oi["open_interest_value"] > 0))

    raw = pd.DataFrame(index=grid.index)
    raw["funding"] = asof_values(close_times, funding, ["funding_mean_3"], pd.Timedelta(hours=9))["funding_mean_3"]
    raw["oi_growth"] = log_oi - log_oi.shift(168)
    raw["crowd_long"] = np.log(oi["global_account_ratio"].where(oi["global_account_ratio"] > 0))
    raw["price_stretch"] = close / close.rolling(720, min_periods=720).mean() - 1
    raw["fear_greed"] = asof_values(close_times, data["fear_greed"], ["fear_greed"], pd.Timedelta(hours=48))["fear_greed"]

    frame = grid.copy()
    for name in COMPONENTS:
        frame[name] = raw[name]
        frame[f"z_{name}"] = _zscore(raw[name])
    z_columns = [f"z_{name}" for name in COMPONENTS]
    frame["euphoria"] = frame[z_columns].mean(axis=1, skipna=False)
    return frame.replace([np.inf, -np.inf], np.nan)
