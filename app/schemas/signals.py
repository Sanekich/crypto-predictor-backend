from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SignalModelQuality(BaseModel):
    """How the model's high-confidence signals performed on the held-out test period."""

    test_period_start: datetime
    test_period_end: datetime
    test_accuracy: float
    always_up_accuracy: float = Field(description="Accuracy of always predicting UP")
    always_down_accuracy: float = Field(default=0.0, description="Accuracy of always predicting DOWN")
    majority_baseline_accuracy: float = Field(default=0.0, description="Accuracy of always predicting the most common direction")
    max_confidence: float = Field(description="Highest calibrated confidence the model reached on the test period")
    signals_at_threshold: int = Field(description="Independent test signals at or above the signal threshold")
    hit_rate: float | None
    hit_rate_lower_95: float | None


class HorizonSignal(BaseModel):
    horizon: str
    model_version: str
    status: Literal["approved", "shadow"] = Field(
        description="approved: signals passed verification; shadow: shown for observation only"
    )
    status_reasons: list[str]
    p_up: float = Field(description="Calibrated probability that the price is higher at target_time")
    direction: Literal["up", "down"]
    confidence: float = Field(description="max(p_up, 1 - p_up)")
    is_signal: bool = Field(description="confidence >= requested threshold and the model has walk-forward skill")
    has_skill: bool = Field(default=False, description="beats the base rate on most walk-forward folds")
    walk_forward_skill: float | None = Field(default=None, description="mean Brier skill vs base rate")
    walk_forward_positive_folds: int | None = None
    walk_forward_folds: int | None = None
    target_time: datetime
    quality: SignalModelQuality


class JournalEntry(BaseModel):
    horizon: str
    signal_time: datetime
    target_time: datetime
    direction: Literal["up", "down"]
    confidence: float
    price_at_signal: float
    outcome_price: float | None = None
    correct: bool | None = Field(default=None, description="None while the horizon has not passed yet")
    model_status: str


class LiveStats(BaseModel):
    signals: int
    resolved: int
    correct: int
    hit_rate: float | None


class SignalsResponse(BaseModel):
    symbol: str
    generated_at: datetime
    data_as_of: datetime
    threshold: float
    horizons: list[HorizonSignal]
    history: list[JournalEntry]
    live: LiveStats
    disclaimer: str = "Research signals. Not financial advice."
