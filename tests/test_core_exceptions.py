import io
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from src.core.exceptions import (
    ConflictError,
    DocumentParsingError,
    LLMProviderError,
    NotFoundError,
)


class TestExceptionClasses:
    """Проверка, что кастомные исключения хранят detail."""

    def test_not_found_default(self):
        exc = NotFoundError()
        assert exc.detail == "Resource not found"
        assert str(exc) == "Resource not found"

    def test_not_found_custom(self):
        exc = NotFoundError("User not found")
        assert exc.detail == "User not found"

    def test_conflict_default(self):
        exc = ConflictError()
        assert exc.detail == "Conflict"

    def test_conflict_custom(self):
        exc = ConflictError("Email already exists")
        assert exc.detail == "Email already exists"

    def test_document_parsing_default(self):
        exc = DocumentParsingError()
        assert exc.detail == "Failed to parse document"

    def test_document_parsing_custom(self):
        exc = DocumentParsingError("Unsupported content type: image/png")
        assert exc.detail == "Unsupported content type: image/png"

    def test_llm_provider_default(self):
        exc = LLMProviderError()
        assert exc.detail == "LLM provider error"

    def test_llm_provider_custom(self):
        exc = LLMProviderError("OpenAI API timeout")
        assert exc.detail == "OpenAI API timeout"

    def test_all_inherit_from_exception(self):
        for cls in (NotFoundError, ConflictError, DocumentParsingError, LLMProviderError):
            assert issubclass(cls, Exception)

    def test_exceptions_are_raisable(self):
        """Все исключения могут быть выброшены и пойманы."""
        for cls in (NotFoundError, ConflictError, DocumentParsingError, LLMProviderError):
            with pytest.raises(cls):
                raise cls("test")


@pytest.mark.asyncio
class TestExceptionHandlers:
    """Интеграционные тесты: exception handlers в main.py возвращают правильные HTTP-коды."""

    async def test_not_found_returns_404(self, client: AsyncClient):
        resp = await client.get("/users/999")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "User not found"

    async def test_not_found_document_returns_404(self, client: AsyncClient):
        resp = await client.get("/documents/999")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Document not found"

    async def test_delete_not_found_returns_404(self, client: AsyncClient):
        resp = await client.delete("/users/999")
        assert resp.status_code == 404

    async def test_delete_document_not_found_returns_404(self, client: AsyncClient):
        resp = await client.delete("/documents/999")
        assert resp.status_code == 404

    async def test_document_parsing_error_returns_422(self, client: AsyncClient):
        """Upload файла с неподдерживаемым типом → DocumentParsingError → 422."""
        user_resp = await client.post("/users/", json={"name": "TestUser"})
        user_id = user_resp.json()["id"]

        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("test.bin", io.BytesIO(b"binary"), "application/octet-stream")},
        )
        assert resp.status_code == 422
        assert "Unsupported content type" in resp.json()["detail"]

    async def test_invalid_path_param_returns_422(self, client: AsyncClient):
        """Нечисловой ID в path → 422 от FastAPI."""
        resp = await client.get("/documents/abc")
        assert resp.status_code == 422

    async def test_negative_id_returns_422(self, client: AsyncClient):
        """Отрицательный ID → 422 (gt=0 валидация)."""
        resp = await client.get("/documents/-1")
        assert resp.status_code == 422

    async def test_zero_id_returns_422(self, client: AsyncClient):
        """Нулевой ID → 422 (gt=0 валидация)."""
        resp = await client.get("/documents/0")
        assert resp.status_code == 422

    async def test_sqlalchemy_error_returns_500(self, client: AsyncClient):
        """SQLAlchemyError → 500 через exception handler."""
        from sqlalchemy.exc import SQLAlchemyError

        with patch(
            "src.modules.users.services.UserService.get_by_id",
            side_effect=SQLAlchemyError("DB connection lost"),
        ):
            resp = await client.get("/users/1")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"

    async def test_generic_exception_returns_500(self):
        """Непредвиденное исключение → 500 через generic handler."""
        from starlette.requests import Request
        from src.main import generic_error_handler

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "query_string": b"",
        }
        request = Request(scope)
        response = await generic_error_handler(request, RuntimeError("unexpected crash"))
        assert response.status_code == 500
        body = response.body.decode()
        assert "Internal server error" in body

    async def test_conflict_error_returns_409(self, client: AsyncClient):
        """ConflictError → 409 через exception handler."""
        with patch(
            "src.modules.users.services.UserService.create",
            side_effect=ConflictError("User already exists"),
        ):
            resp = await client.post("/users/", json={"name": "Alice"})
        assert resp.status_code == 409
        assert resp.json()["detail"] == "User already exists"

    async def test_llm_provider_error_returns_502(self):
        """LLMProviderError → 502 через exception handler."""
        from starlette.requests import Request
        from src.main import llm_provider_handler

        scope = {"type": "http", "method": "POST", "path": "/rag/search"}
        request = Request(scope)
        response = await llm_provider_handler(request, LLMProviderError("Gemini API timeout"))
        assert response.status_code == 502
        body = response.body.decode()
        assert "Gemini API timeout" in body
