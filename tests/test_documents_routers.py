import io

import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


class TestDocumentsAPI:
    async def _create_user(self, client: AsyncClient) -> int:
        resp = await client.post("/users/", json={"name": "TestUser"})
        return resp.json()["id"]

    # ── Upload ────────────────────────────────────────────

    async def test_upload_txt_document(self, client: AsyncClient):
        user_id = await self._create_user(client)

        file_content = b"Hello, this is a test document."
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "test.txt"
        assert data["content_type"] == "text/plain"
        assert data["status"] == "ready"

    async def test_upload_without_user_id(self, client: AsyncClient):
        """user_id — обязательный query-параметр."""
        resp = await client.post(
            "/documents/upload",
            files={"file": ("test.txt", io.BytesIO(b"data"), "text/plain")},
        )
        assert resp.status_code == 422

    async def test_upload_without_file(self, client: AsyncClient):
        """Загрузка без файла → 422."""
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
        )
        assert resp.status_code == 422

    async def test_upload_unsupported_type(self, client: AsyncClient):
        """Неподдерживаемый content_type → 422 (DocumentParsingError)."""
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("img.png", io.BytesIO(b"\x89PNG"), "image/png")},
        )
        assert resp.status_code == 422
        assert "Unsupported content type" in resp.json()["detail"]

    async def test_upload_negative_user_id(self, client: AsyncClient):
        """Отрицательный user_id → 422."""
        resp = await client.post(
            "/documents/upload",
            params={"user_id": -1},
            files={"file": ("t.txt", io.BytesIO(b"x"), "text/plain")},
        )
        assert resp.status_code == 422

    # ── Get list ──────────────────────────────────────────

    async def test_get_documents(self, client: AsyncClient):
        user_id = await self._create_user(client)

        await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("a.txt", io.BytesIO(b"aaa"), "text/plain")},
        )
        await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("b.txt", io.BytesIO(b"bbb"), "text/plain")},
        )

        resp = await client.get("/documents/", params={"user_id": user_id})
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_get_documents_empty(self, client: AsyncClient):
        user_id = await self._create_user(client)
        resp = await client.get("/documents/", params={"user_id": user_id})
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_documents_without_user_id(self, client: AsyncClient):
        """user_id — обязательный query-параметр."""
        resp = await client.get("/documents/")
        assert resp.status_code == 422

    async def test_get_documents_nonexistent_user(self, client: AsyncClient):
        """Несуществующий user_id → пустой список (не ошибка)."""
        resp = await client.get("/documents/", params={"user_id": 999})
        assert resp.status_code == 200
        assert resp.json() == []

    # ── Get by ID ─────────────────────────────────────────

    async def test_get_document_by_id(self, client: AsyncClient):
        user_id = await self._create_user(client)

        upload_resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
        )
        doc_id = upload_resp.json()["id"]

        resp = await client.get(f"/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["filename"] == "test.txt"

    async def test_get_document_not_found(self, client: AsyncClient):
        resp = await client.get("/documents/9999")
        assert resp.status_code == 404

    async def test_get_document_invalid_id_string(self, client: AsyncClient):
        """Строка вместо числа → 422."""
        resp = await client.get("/documents/abc")
        assert resp.status_code == 422

    async def test_get_document_zero_id(self, client: AsyncClient):
        resp = await client.get("/documents/0")
        assert resp.status_code == 422

    async def test_get_document_negative_id(self, client: AsyncClient):
        resp = await client.get("/documents/-5")
        assert resp.status_code == 422

    # ── Delete ────────────────────────────────────────────

    async def test_delete_document(self, client: AsyncClient):
        user_id = await self._create_user(client)

        upload_resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
        )
        doc_id = upload_resp.json()["id"]

        resp = await client.delete(f"/documents/{doc_id}")
        assert resp.status_code == 204

        get_resp = await client.get(f"/documents/{doc_id}")
        assert get_resp.status_code == 404

    async def test_delete_document_not_found(self, client: AsyncClient):
        resp = await client.delete("/documents/9999")
        assert resp.status_code == 404

    async def test_delete_document_zero_id(self, client: AsyncClient):
        resp = await client.delete("/documents/0")
        assert resp.status_code == 422

    async def test_delete_document_negative_id(self, client: AsyncClient):
        resp = await client.delete("/documents/-1")
        assert resp.status_code == 422

    # ── Response shape ────────────────────────────────────

    async def test_upload_response_shape(self, client: AsyncClient):
        """Ответ upload содержит все поля DocumentRead."""
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("doc.txt", io.BytesIO(b"data"), "text/plain")},
        )
        data = resp.json()
        assert "id" in data
        assert "filename" in data
        assert "content_type" in data
        assert "status" in data
        assert "created_at" in data
