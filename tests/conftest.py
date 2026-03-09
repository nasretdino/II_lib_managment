import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.base_model import Base
from src.db import get_db
from src.main import app
from src.modules.documents.storage import get_object_storage
from src.modules.rag.dependencies import get_rag
from src.modules.rag.schemas import SearchResult, IndexResult


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine_test = create_async_engine(TEST_DATABASE_URL, echo=False)

async_session_test = async_sessionmaker(
    bind=engine_test,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


LIGHTRAG_ENV_KEYS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DATABASE",
    "POSTGRES_WORKSPACE",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "NEO4J_WORKSPACE",
)


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Создаёт таблицы перед каждым тестом и удаляет после."""
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine_test.dispose()


def _mock_object_storage():
    mock = MagicMock()
    mock.upload_bytes = AsyncMock(return_value="s3://documents/users/1/mock-file.txt")
    mock.delete_by_uri = AsyncMock()
    mock.ensure_bucket = AsyncMock()
    return mock


@pytest.fixture(autouse=True)
def isolate_rag_storage(tmp_path):
    """Перенаправляет RAG storage в tmp, чтобы тесты не создавали rag_storage/ в репозитории."""
    test_rag_storage = tmp_path / "rag_storage"
    with patch("src.modules.rag.services.WORKING_DIR", test_rag_storage):
        yield test_rag_storage


@pytest.fixture(autouse=True)
def isolate_lightrag_env():
    """Восстанавливает env LightRAG после каждого теста, чтобы избежать межтестовых утечек."""
    old_values = {key: os.environ.get(key) for key in LIGHTRAG_ENV_KEYS}
    yield
    for key, old_value in old_values.items():
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Сессия БД для unit-тестов DAO/Service."""
    async with async_session_test() as session:
        yield session
        await session.rollback()


async def _override_get_db():
    async with async_session_test() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _mock_rag_service():
    """Мок RagService — не требует Gemini API в тестах."""
    mock = MagicMock()
    mock.index_document = AsyncMock(
        return_value=IndexResult(doc_id=0, chunks_count=0, status="indexed")
    )
    mock.search = AsyncMock(
        return_value=SearchResult(context_text="mock context", sources=[], mode="hybrid")
    )
    mock.delete_document = AsyncMock()
    mock.initialize = AsyncMock()
    mock.shutdown = AsyncMock()
    return mock


@pytest_asyncio.fixture
async def client():
    """HTTP-клиент для интеграционных тестов роутеров."""
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_rag] = _mock_rag_service
    app.dependency_overrides[get_object_storage] = _mock_object_storage
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
