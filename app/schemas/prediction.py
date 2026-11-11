from datetime import datetime

from pydantic import BaseModel, Field


class PriceRange(BaseModel):
    """Forecast range: with `probability`, the price at the horizon is expected
    to end between lower_price and upper_price."""

    probability: float = Field(description="Nominal probability of the range, e.g. 0.8")
    lower_quantile: float
    upper_quantile: float
    lower_return: float = Field(description="Lower bound as a return, e.g. -0.025 == -2.5%")
    upper_return: float
    lower_price: float
    upper_price: float


class DirectionForecast(BaseModel):
    """Calibrated direction forecast for the requested horizon."""

    available: bool
    up_probability: float | None = None
    direction: str | None = Field(default=None, description="up or down")
    confidence: float | None = None
    status: str | None = Field(default=None, description="approved or shadow")
    model_version: str | None = None
    reason: str | None = Field(default=None, description="NO_SIGNAL | NO_MODEL | SHADOW")


class ModelQuality(BaseModel):
    """Out-of-sample quality of the range model on the held-out test period."""

    test_period_start: datetime
    test_period_end: datetime
    nominal_coverage: float
    test_coverage: float = Field(description="Share of test outcomes that fell inside the range")
    improvement_vs_baseline: float = Field(description="Relative pinball-loss improvement over a constant historical range")


class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    horizon: str
    generated_at: datetime
    data_as_of: datetime = Field(description="Close time of the last closed candle used")
    target_time: datetime = Field(description="Time the forecast refers to")
    model_version: str
    current_price: float
    median_return: float
    median_price: float
    range: PriceRange
    direction: DirectionForecast
    quality: ModelQuality
    disclaimer: str = "Research forecast. Not financial advice."


class AssetInfo(BaseModel):
    symbol: str
    name: str
    timeframe: str
    horizons: list[str]


class ModelSummary(BaseModel):
    model_id: str
    model_type: str
    symbol: str
    interval: str
    horizon: str
    trained_at: datetime
