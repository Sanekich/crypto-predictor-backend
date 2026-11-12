"""Features from derivatives history (funding, basis, open interest & positioning, DVOL).

Leakage rule: a candle that opens at t closes at t + interval. Its features may
only use derivatives rows with available_at <= t + interval. Values are attached
with an as-of join on that moment, with a tolerance so that a long outage of a
source produces NaN instead of silently repeating an old value.

All features are scale-free (rates, ratios, log changes).
"""

import numpy as np
import pandas as pd

from app.features.returns import log_return

GROUPS = ("funding", "basis", "open_interest", "dvol")

# dataset in DerivativesStore needed by each feature group
GROUP_DATASET = {"funding": "funding", "basis": "premium", "open_interest": "metrics", "dvol": "dvol"}


def _ns(series: pd.Series) -> pd.Series:
    return series.dt.as_unit("ns")


def asof_values(
    close_times: pd.Series,
    frame: pd.DataFrame,
    columns: list[str],
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """For each close time: the latest row with available_at <= close time (within tolerance)."""
    left = pd.DataFrame({"close_time": _ns(close_times)}).reset_index()
    right = frame[["available_at", *columns]].dropna(subset=["available_at"]).copy()
    right["available_at"] = _ns(right["available_at"])
    right = right.sort_values("available_at")
    merged = pd.merge_asof(
        left.sort_values("close_time"),
        right,
        left_on="close_time",
        right_on="available_at",
        direction="backward",
        tolerance=tolerance,
    )
    return merged.sort_values("index").set_index("index")[columns].set_axis(close_times.index)


def _hourly_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """5-minute snapshots -> one row per hour (last level, mean taker ratio), available at the hour end."""
    m = metrics.copy()
    m["hour_end"] = _ns(m["available_at"]).dt.ceil("h")
    hourly = m.groupby("hour_end").agg(
        open_interest_value=("open_interest_value", "last"),
        top_trader_position_ratio=("top_trader_position_ratio", "last"),
        global_account_ratio=("global_account_ratio", "last"),
        taker_buy_sell_ratio=("taker_buy_sell_ratio", "mean"),
        snapshots=("available_at", "size"),
    )
    hourly = hourly[hourly["snapshots"] >= 6]  # at least half of the hour observed
    return hourly.reset_index().rename(columns={"hour_end": "available_at"})


def build_derivative_features(
    grid: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    groups: tuple[str, ...],
    interval_delta: pd.Timedelta,
) -> pd.DataFrame:
    """Derivative features aligned to `grid` (columns timestamp, close; one row per candle)."""
    missing = [g for g in groups if GROUP_DATASET[g] not in data]
    if missing:
        raise ValueError(f"derivatives data missing for: {', '.join(missing)}")

    close_times = grid["timestamp"] + interval_delta
    close = grid["close"]
    f: dict[str, pd.Series] = {}

    if "funding" in groups:
        funding = data["funding"].sort_values("available_at").copy()
        funding["funding_mean_3"] = funding["funding_rate"].rolling(3, min_periods=3).mean()  # last 24 h
        funding["funding_mean_21"] = funding["funding_rate"].rolling(21, min_periods=21).mean()  # last 7 days
        values = asof_values(close_times, funding, ["funding_rate", "funding_mean_3", "funding_mean_21"], pd.Timedelta(hours=9))
        f["funding_rate"] = values["funding_rate"]
        f["funding_mean_3"] = values["funding_mean_3"]
        f["funding_mean_21"] = values["funding_mean_21"]

    if "basis" in groups:
        basis = asof_values(close_times, data["premium"], ["premium_close"], pd.Timedelta(minutes=30))["premium_close"]
        mean_168 = basis.rolling(168, min_periods=168).mean()
        std_168 = basis.rolling(168, min_periods=168).std()
        f["basis"] = basis
        f["basis_mean_24"] = basis.rolling(24, min_periods=24).mean()
        f["basis_change_24"] = basis - basis.shift(24)
        f["basis_zscore_168"] = (basis - mean_168) / std_168.where(std_168 > 0)

    if "open_interest" in groups:
        hourly = _hourly_metrics(data["metrics"])
        values = asof_values(
            close_times,
            hourly,
            ["open_interest_value", "top_trader_position_ratio", "global_account_ratio", "taker_buy_sell_ratio"],
            pd.Timedelta(minutes=30),
        )
        log_oi = np.log(values["open_interest_value"].where(values["open_interest_value"] > 0))
        f["oi_change_1"] = log_oi - log_oi.shift(1)
        f["oi_change_24"] = log_oi - log_oi.shift(24)
        f["oi_price_divergence_24"] = f["oi_change_24"] - log_return(close, 24)
        f["top_trader_ratio_log"] = np.log(values["top_trader_position_ratio"].where(values["top_trader_position_ratio"] > 0))
        f["global_account_ratio_log"] = np.log(values["global_account_ratio"].where(values["global_account_ratio"] > 0))
        taker = np.log(values["taker_buy_sell_ratio"].where(values["taker_buy_sell_ratio"] > 0))
        f["taker_ratio_log_mean_24"] = taker.rolling(24, min_periods=24).mean()

    if "dvol" in groups:
        dvol = asof_values(close_times, data["dvol"], ["dvol"], pd.Timedelta(minutes=30))["dvol"]
        realized_hourly = log_return(close, 1).rolling(24, min_periods=24).std()
        implied_hourly = dvol / 100 / np.sqrt(24 * 365)
        f["dvol"] = dvol / 100
        f["dvol_change_24"] = np.log(dvol / dvol.shift(24))
        f["dvol_to_realized_24"] = implied_hourly / realized_hourly.where(realized_hourly > 0)

    return pd.DataFrame(f, index=grid.index).replace([np.inf, -np.inf], np.nan)
