from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CryptoPredictor"
    app_env: str = "development"
    debug: bool = False
    app_version: str = "0.1.0"

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    database_url: str | None = None

    market_data_provider: str = "binance"
    market_data_api_key: str | None = None
    market_data_api_secret: str | None = None

    model_artifacts_dir: Path = Path("artifacts")


@lru_cache
def get_settings() -> Settings:
    return Settings()
