from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import barrier, data_status, health, live, models, operations, predictions, pytorch, signals, training_stage
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
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
    app.include_router(pytorch.router)
    return app


app = create_app()
