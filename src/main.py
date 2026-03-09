from contextlib import asynccontextmanager
import time
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from starlette.requests import Request

from src.core import (
    settings,
    setup_logging,
    ConflictError,
    DailyQuotaExhaustedError,
    DocumentParsingError,
    LLMProviderError,
    NotFoundError,
)
from src.db import engine
from src.modules import user_router, document_router, rag_router
from src.modules.rag.services import get_rag_service

# ── Logging ───────────────────────────────────────────────
setup_logging(settings.env)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация LightRAG при старте
    rag = get_rag_service()
    await rag.initialize()
    logger.info("LightRAG initialized")

    yield

    # Корректное завершение LightRAG
    await rag.shutdown()
    await engine.dispose()


app = FastAPI(
    lifespan=lifespan,
    debug=settings.env == "dev",
    docs_url="/docs" if settings.env != "prod" else None,
    redoc_url="/redoc" if settings.env != "prod" else None,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id

    bound_logger = logger.bind(request_id=request_id)
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        bound_logger.exception(
            "HTTP request failed: {} {} in {:.2f}ms",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - started_at) * 1000
    bound_logger.info(
        "HTTP {} {} -> {} ({:.2f}ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


def _response_with_request_id(
    request: Request,
    status_code: int,
    detail: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = headers.copy() if headers else {}
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response_headers["X-Request-ID"] = request_id
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=response_headers)


def _request_meta(request: Request) -> tuple[str, str]:
    method = request.scope.get("method", "-")
    path = request.scope.get("path", "-")
    return method, path


# ── Exception handlers ────────────────────────────────────
@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    method, path = _request_meta(request)
    logger.bind(request_id=getattr(request.state, "request_id", "-")).warning(
        "NotFoundError on {} {}: {}",
        method,
        path,
        exc.detail,
    )
    return _response_with_request_id(request, 404, exc.detail)


@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError):
    method, path = _request_meta(request)
    logger.bind(request_id=getattr(request.state, "request_id", "-")).warning(
        "ConflictError on {} {}: {}",
        method,
        path,
        exc.detail,
    )
    return _response_with_request_id(request, 409, exc.detail)


@app.exception_handler(DocumentParsingError)
async def document_parsing_handler(request: Request, exc: DocumentParsingError):
    method, path = _request_meta(request)
    logger.bind(request_id=getattr(request.state, "request_id", "-")).warning(
        "DocumentParsingError on {} {}: {}",
        method,
        path,
        exc.detail,
    )
    return _response_with_request_id(request, 422, exc.detail)


@app.exception_handler(DailyQuotaExhaustedError)
async def daily_quota_handler(request: Request, exc: DailyQuotaExhaustedError):
    method, path = _request_meta(request)
    logger.bind(request_id=getattr(request.state, "request_id", "-")).warning(
        "DailyQuotaExhaustedError on {} {}",
        method,
        path,
    )
    return _response_with_request_id(
        request,
        429,
        exc.detail,
        headers={"Retry-After": "3600"},
    )


@app.exception_handler(LLMProviderError)
async def llm_provider_handler(request: Request, exc: LLMProviderError):
    method, path = _request_meta(request)
    logger.bind(request_id=getattr(request.state, "request_id", "-")).error(
        "LLMProviderError on {} {}: {}",
        method,
        path,
        exc.detail,
    )
    return _response_with_request_id(request, 502, exc.detail)


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    method, path = _request_meta(request)
    logger.bind(request_id=getattr(request.state, "request_id", "-")).error(
        "Database error on {} {}: {}",
        method,
        path,
        exc,
    )
    return _response_with_request_id(request, 500, "Internal server error")


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    method, path = _request_meta(request)
    logger.bind(request_id=getattr(request.state, "request_id", "-")).exception(
        "Unhandled error on {} {}",
        method,
        path,
    )
    return _response_with_request_id(request, 500, "Internal server error")


# ── Routers ───────────────────────────────────────────────
app.include_router(user_router)
app.include_router(document_router)
app.include_router(rag_router)

