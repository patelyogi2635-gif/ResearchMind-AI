import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Domain error that maps to a clean, client-safe JSON response.

    Attributes:
        message: Client-safe description of what went wrong.
        status_code: HTTP status returned to the client.
        code: Stable machine-readable error code.
        details: Optional extra structured context that is safe to expose.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "internal_error",
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details


def _request_id(request: Request) -> str:
    """Returns the request id stored by the request-context middleware."""
    return getattr(request.state, "request_id", "-")


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
) -> JSONResponse:
    """Builds the standard error envelope.

    The envelope keeps a top-level "detail" string so existing clients that
    read `detail` keep working.
    """
    request_id = _request_id(request)
    body = {
        "detail": message,
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": details,
        },
    }
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={"X-Request-ID": request_id},
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handles AppError raised anywhere in the request lifecycle."""
    log = logger.error if exc.status_code >= 500 else logger.warning
    log(
        "app_error code=%s status=%s message=%s",
        exc.code,
        exc.status_code,
        exc.message,
        exc_info=exc.__cause__,
    )
    return _error_response(request, exc.status_code, exc.code, exc.message, exc.details)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handles request validation failures with a compact, readable message."""
    issues = [
        {
            "field": ".".join(str(part) for part in error["loc"] if part != "body"),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    message = "Invalid request: " + "; ".join(
        f"{issue['field']}: {issue['message']}" for issue in issues
    )
    logger.warning("validation_error issues=%s", issues)
    return _error_response(request, 422, "validation_error", message, issues)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handles framework HTTP errors such as 404 and 405."""
    return _error_response(request, exc.status_code, f"http_{exc.status_code}", str(exc.detail))


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handles unexpected exceptions without leaking internals to the client."""
    logger.error(
        "unhandled_error request_id=%s path=%s",
        _request_id(request),
        request.url.path,
        exc_info=exc,
    )
    return _error_response(request, 500, "internal_error", "Internal server error")


def register_exception_handlers(app: FastAPI) -> None:
    """Registers all exception handlers on the application.

    Args:
        app: The FastAPI application.
    """
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
