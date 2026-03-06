from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from starlette.requests import Request

from src.core import (
    settings,
    setup_logging,
    ConflictError,
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


# ── Exception handlers ────────────────────────────────────
@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": exc.detail})


@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError):
    return JSONResponse(status_code=409, content={"detail": exc.detail})


@app.exception_handler(DocumentParsingError)
async def document_parsing_handler(request: Request, exc: DocumentParsingError):
    return JSONResponse(status_code=422, content={"detail": exc.detail})


@app.exception_handler(LLMProviderError)
async def llm_provider_handler(request: Request, exc: LLMProviderError):
    return JSONResponse(status_code=502, content={"detail": exc.detail})


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    logger.error("Database error on {} {}: {}", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on {} {}", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Routers ───────────────────────────────────────────────
app.include_router(user_router)
app.include_router(document_router)
app.include_router(rag_router)

