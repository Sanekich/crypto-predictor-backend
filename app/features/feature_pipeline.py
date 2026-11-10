"""Builds model features and targets from a processed OHLCV dataset.

Rules (architecture doc, section 6.6):
- features at row t use only candles <= t (the candle t is closed);
- features are scale-free (ratios, returns) instead of raw price levels,
  so a model trained at 10k$ BTC is applicable at 100k$ BTC;
- the feature list and its order are deterministic and saved with the dataset;
- no fitted transformations (scalers) here: they belong to model training,
  where they are fitted on the training split only.

Window sizes are in candles. The defaults are chosen for 1h candles.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.data.processing.resampling import reindex_to_grid
from app.features import returns, technical_indicators as ti, volatility, volume
from app.features.targets import future_direction, future_return, horizon_to_periods

FEATURE_PIPELINE_VERSION = "1.0"

NON_FEATURE_COLUMNS = ("timestamp", "close")


@dataclass(frozen=True)
class FeatureConfig:
    return_windows: tuple[int, ...] = (1, 3, 6, 12, 24, 72, 168)
    sma_windows: tuple[int, ...] = (24, 72, 168)
    sma_change_window: int = 24
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    rsi_window: int = 14
    atr_window: int = 14
    volatility_windows: tuple[int, ...] = (24, 168)
    volume_windows: tuple[int, ...] = (24, 168)
    time_features: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_features(ohlcv: pd.DataFrame, interval: str, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Return a frame with `timestamp`, `close` and feature columns on a regular grid.

    Rows for missing candles (and rows whose windows touch them) contain NaN.
    """
    config = config or FeatureConfig()
    df = reindex_to_grid(ohlcv, interval)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    f: dict[str, pd.Series] = {}

    # Returns
    for w in config.return_windows:
        f[f"log_return_{w}"] = returns.log_return(close, w)

    # Moving averages (distance of price from the average, and its slope)
    for w in config.sma_windows:
        f[f"close_to_sma_{w}"] = close / ti.sma(close, w) - 1
    first_sma = ti.sma(close, config.sma_windows[0])
    f[f"sma_{config.sma_windows[0]}_change_{config.sma_change_window}"] = (
        first_sma / first_sma.shift(config.sma_change_window) - 1
    )
    ema_fast = ti.ema(close, config.ema_fast)
    ema_slow = ti.ema(close, config.ema_slow)
    f[f"close_to_ema_{config.ema_fast}"] = close / ema_fast - 1
    f[f"ema_{config.ema_fast}_to_ema_{config.ema_slow}"] = ema_fast / ema_slow - 1

    # Momentum
    f[f"rsi_{config.rsi_window}"] = ti.rsi(close, config.rsi_window)
    macd_line, macd_signal, macd_hist = ti.macd(close, config.ema_fast, config.ema_slow, config.macd_signal)
    f["macd_pct"] = macd_line / close
    f["macd_signal_pct"] = macd_signal / close
    f["macd_hist_pct"] = macd_hist / close

    # Volatility
    for w in config.volatility_windows:
        f[f"volatility_{w}"] = volatility.rolling_volatility(close, w)
    if len(config.volatility_windows) >= 2:
        short_w, long_w = config.volatility_windows[0], config.volatility_windows[-1]
        f[f"volatility_ratio_{short_w}_{long_w}"] = f[f"volatility_{short_w}"] / f[f"volatility_{long_w}"]
    f["candle_range_pct"] = volatility.candle_range(high, low, close)
    f[f"atr_{config.atr_window}_pct"] = ti.atr(high, low, close, config.atr_window) / close

    # Volume
    f["volume_change_1"] = volume.volume_change(vol, 1)
    for w in config.volume_windows:
        f[f"volume_ratio_{w}"] = volume.volume_ratio(vol, w)
    f[f"volume_zscore_{config.volume_windows[-1]}"] = volume.volume_zscore(vol, config.volume_windows[-1])
    if "taker_buy_base" in df.columns:
        f["taker_buy_ratio"] = volume.taker_buy_ratio(df["taker_buy_base"], vol)

    # Calendar (candle open time, UTC); known in advance, no leakage
    if config.time_features:
        hour = df["timestamp"].dt.hour
        weekday = df["timestamp"].dt.day_of_week
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        f["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        f["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        # Calendar values exist even for missing candles; blank them for consistency
        for name in ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos"):
            f[name] = f[name].where(close.notna())

    features = pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)
    return pd.concat([df[["timestamp", "close"]], features], axis=1)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in NON_FEATURE_COLUMNS]


@dataclass
class FeatureDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_return_column: str
    target_direction_column: str
    horizon: str
    horizon_periods: int
    interval: str
    config: FeatureConfig
    rows_before_dropna: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        ts = self.frame["timestamp"]
        return {
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "interval": self.interval,
            "horizon": self.horizon,
            "horizon_periods": self.horizon_periods,
            "rows": len(self.frame),
            "rows_dropped": self.rows_before_dropna - len(self.frame),
            "first_timestamp": ts.iloc[0].isoformat() if len(ts) else None,
            "last_timestamp": ts.iloc[-1].isoformat() if len(ts) else None,
            "feature_columns": self.feature_columns,
            "target_return_column": self.target_return_column,
            "target_direction_column": self.target_direction_column,
            "positive_target_share": float(self.frame[self.target_direction_column].mean()) if len(ts) else None,
            "config": self.config.to_dict(),
            **self.metadata,
        }


def build_dataset(
    ohlcv: pd.DataFrame,
    interval: str,
    horizon: str,
    config: FeatureConfig | None = None,
) -> FeatureDataset:
    """Features + targets for training. Rows with any missing feature or target are dropped.

    The last `horizon_periods` rows have no known future and are therefore
    excluded; use `build_features` for inference on the latest candle.
    """
    config = config or FeatureConfig()
    periods = horizon_to_periods(horizon, interval)
    frame = build_features(ohlcv, interval, config)
    columns = feature_columns(frame)

    target_return = f"target_return_{horizon}"
    target_direction = f"target_direction_{horizon}"
    frame[target_return] = future_return(frame["close"], periods)
    frame[target_direction] = future_direction(frame[target_return])

    rows_before = int(frame["close"].notna().sum())
    clean = frame.dropna(subset=[*columns, target_return]).reset_index(drop=True)

    return FeatureDataset(
        frame=clean,
        feature_columns=columns,
        target_return_column=target_return,
        target_direction_column=target_direction,
        horizon=horizon,
        horizon_periods=periods,
        interval=interval,
        config=config,
        rows_before_dropna=rows_before,
    )
