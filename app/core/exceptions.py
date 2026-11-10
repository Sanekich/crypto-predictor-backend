"""Application errors with a stable code, returned to the client as
{"detail": {"code": "...", "message": "..."}}."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


class ModelNotAvailableError(AppError):
    status_code = 404
    code = "MODEL_NOT_AVAILABLE"


class DataNotAvailableError(AppError):
    status_code = 503
    code = "DATA_NOT_AVAILABLE"


class DataStaleError(AppError):
    status_code = 503
    code = "DATA_STALE"


class ModelIncompatibleError(AppError):
    status_code = 500
    code = "MODEL_INCOMPATIBLE"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )
