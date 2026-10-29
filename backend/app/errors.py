"""RFC 9457 Problem Details for every error response."""

from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = structlog.get_logger()
PROBLEM = "application/problem+json"


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        detail: str | None = None,
        *,
        errors: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self.status, self.code, self.detail = status, code, detail
        self.errors, self.headers, self.extra = errors, headers, extra or {}


def not_found(what: str = "Resource") -> ApiError:
    return ApiError(404, "not_found", f"{what} not found")


def forbidden(detail: str = "You don't have permission to do this") -> ApiError:
    return ApiError(403, "forbidden", detail)


def problem(
    request: Request, status: int, code: str, detail: str | None, errors=None, headers=None, extra=None
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"https://errors.automl.local/{code}",
        "title": HTTPStatus(status).phrase,
        "status": status,
        "detail": detail or HTTPStatus(status).description,
        "instance": request.url.path,
        "code": code,
    }
    if errors:
        body["errors"] = errors
    body.update(extra or {})
    return JSONResponse(body, status_code=status, media_type=PROBLEM, headers=headers)


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(request: Request, exc: ApiError):
        return problem(request, exc.status, exc.code, exc.detail, exc.errors, exc.headers, exc.extra)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return problem(request, 422, "validation_error", "Request validation failed", errors)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return problem(request, exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled_error", path=request.url.path)
        return problem(request, 500, "internal_error", "Something went wrong on our side")
