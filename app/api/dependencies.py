"""Shared service instances for API routes (overridable in tests)."""

from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.core.versioning import PROJECT_ROOT
from app.data.collection.binance_client import BinanceClient
from app.data.storage import MarketDataStorage
from app.ml.registry.model_registry import ModelRegistry
from app.services.market_data_service import MarketDataService
from app.services.prediction_service import PredictionService
from app.services.signal_journal import SignalJournal
from app.services.signal_service import SignalService


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_registry() -> ModelRegistry:
    return ModelRegistry(_resolve(get_settings().model_artifacts_dir))


@lru_cache
def get_market_data_service() -> MarketDataService:
    settings = get_settings()
    return MarketDataService(
        storage=MarketDataStorage(_resolve(settings.data_dir)),
        source_factory=lambda: BinanceClient(base_url=settings.binance_base_url),
        provider=settings.market_data_provider,
    )


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService(registry=get_registry(), market_data=get_market_data_service())


@lru_cache
def get_signal_service() -> SignalService:
    journal = SignalJournal(_resolve(get_settings().data_dir) / "signals")
    return SignalService(registry=get_registry(), market_data=get_market_data_service(), journal=journal)
