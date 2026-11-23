import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import barrier, data_status, euphoria, experiments, health, live, model_history, models, operations, paper_trading, predictions, signals, training_stage, volatility
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Background paper-trading strategies (forward test), one tick per interval."""
    settings = get_settings()
    stop = threading.Event()
    worker = None
    if settings.paper_trading_loop:
        worker = threading.Thread(
            target=paper_trading.strategy_loop, args=(stop, settings.paper_trading_interval_seconds), name="paper-strategies", daemon=True
        )
        worker.start()
    yield
    stop.set()
    if worker is not None:
        worker.join(timeout=5)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)

    @app.get("/", tags=["root"])
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(models.router)
    app.include_router(live.router)
    app.include_router(data_status.router)
    app.include_router(operations.router)
    app.include_router(training_stage.router)
    app.include_router(signals.router)
    app.include_router(barrier.router)
    app.include_router(volatility.router)
    app.include_router(euphoria.router)
    app.include_router(experiments.router)
    app.include_router(model_history.router)
    app.include_router(paper_trading.router)
    return app


app = create_app()
