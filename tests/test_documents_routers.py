import io
from unittest.mock import AsyncMock, patch

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

    # ── Upload: все типы файлов ───────────────────────────

    async def test_upload_pdf_document(self, client: AsyncClient):
        """Загрузка PDF — extract_text мокается, файл принимается."""
        user_id = await self._create_user(client)
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            return_value="Extracted PDF content",
        ):
            resp = await client.post(
                "/documents/upload",
                params={"user_id": user_id},
                files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "report.pdf"
        assert data["content_type"] == "application/pdf"
        assert data["status"] == "ready"

    async def test_upload_docx_document(self, client: AsyncClient):
        """Загрузка DOCX."""
        user_id = await self._create_user(client)
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            return_value="Extracted DOCX content",
        ):
            resp = await client.post(
                "/documents/upload",
                params={"user_id": user_id},
                files={
                    "file": (
                        "doc.docx",
                        io.BytesIO(b"PK\x03\x04"),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "doc.docx"
        assert data["content_type"] == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert data["status"] == "ready"

    async def test_upload_doc_document(self, client: AsyncClient):
        """Загрузка DOC (application/msword)."""
        user_id = await self._create_user(client)
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            return_value="Extracted DOC content",
        ):
            resp = await client.post(
                "/documents/upload",
                params={"user_id": user_id},
                files={
                    "file": (
                        "old.doc",
                        io.BytesIO(b"\xd0\xcf\x11\xe0"),
                        "application/msword",
                    )
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "old.doc"
        assert data["content_type"] == "application/msword"
        assert data["status"] == "ready"

    async def test_upload_csv_unsupported(self, client: AsyncClient):
        """CSV (text/csv) — неподдерживаемый тип."""
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
        )
        assert resp.status_code == 422
        assert "Unsupported content type" in resp.json()["detail"]

    async def test_upload_json_unsupported(self, client: AsyncClient):
        """JSON — неподдерживаемый тип."""
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("data.json", io.BytesIO(b'{"a": 1}'), "application/json")},
        )
        assert resp.status_code == 422
        assert "Unsupported content type" in resp.json()["detail"]

    async def test_upload_xml_unsupported(self, client: AsyncClient):
        """XML — неподдерживаемый тип."""
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("data.xml", io.BytesIO(b"<root/>"), "application/xml")},
        )
        assert resp.status_code == 422

    async def test_upload_html_unsupported(self, client: AsyncClient):
        """HTML — неподдерживаемый тип."""
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("page.html", io.BytesIO(b"<html></html>"), "text/html")},
        )
        assert resp.status_code == 422

    async def test_upload_user_id_zero(self, client: AsyncClient):
        """user_id=0 → 422 (gt=0)."""
        resp = await client.post(
            "/documents/upload",
            params={"user_id": 0},
            files={"file": ("t.txt", io.BytesIO(b"x"), "text/plain")},
        )
        assert resp.status_code == 422

    async def test_upload_user_id_string(self, client: AsyncClient):
        """user_id=abc → 422."""
        resp = await client.post(
            "/documents/upload",
            params={"user_id": "abc"},
            files={"file": ("t.txt", io.BytesIO(b"x"), "text/plain")},
        )
        assert resp.status_code == 422

    async def test_upload_large_txt(self, client: AsyncClient):
        """Большой текстовый файл загружается успешно."""
        user_id = await self._create_user(client)
        large_content = b"A long line of text. " * 5000
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("big.txt", io.BytesIO(large_content), "text/plain")},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "ready"

    async def test_upload_unicode_txt(self, client: AsyncClient):
        """Текстовый файл с юникодом."""
        user_id = await self._create_user(client)
        content = "Привет мир 你好世界 مرحبا".encode("utf-8")
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("unicode.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "ready"

    # ── Upload–List–Delete flow ───────────────────────────

    async def test_upload_then_list_then_delete_flow(self, client: AsyncClient):
        """Загрузить 3 документа, проверить список, удалить один, проверить снова."""
        user_id = await self._create_user(client)

        doc_ids = []
        for name in ("a.txt", "b.txt", "c.txt"):
            resp = await client.post(
                "/documents/upload",
                params={"user_id": user_id},
                files={"file": (name, io.BytesIO(b"content"), "text/plain")},
            )
            assert resp.status_code == 201
            doc_ids.append(resp.json()["id"])

        # Список содержит 3 документа
        resp = await client.get("/documents/", params={"user_id": user_id})
        assert len(resp.json()) == 3

        # Удалить средний
        resp = await client.delete(f"/documents/{doc_ids[1]}")
        assert resp.status_code == 204

        # В списке осталось 2
        resp = await client.get("/documents/", params={"user_id": user_id})
        remaining = resp.json()
        assert len(remaining) == 2
        remaining_ids = [d["id"] for d in remaining]
        assert doc_ids[1] not in remaining_ids

    async def test_delete_then_double_delete(self, client: AsyncClient):
        """Повторное удаление → 404."""
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
        )
        doc_id = resp.json()["id"]

        resp = await client.delete(f"/documents/{doc_id}")
        assert resp.status_code == 204

        resp = await client.delete(f"/documents/{doc_id}")
        assert resp.status_code == 404

    # ── Get list edge cases ───────────────────────────────

    async def test_get_documents_user_id_zero(self, client: AsyncClient):
        """user_id=0 → 422."""
        resp = await client.get("/documents/", params={"user_id": 0})
        assert resp.status_code == 422

    async def test_get_documents_negative_user_id(self, client: AsyncClient):
        """user_id отрицательный → 422."""
        resp = await client.get("/documents/", params={"user_id": -1})
        assert resp.status_code == 422

    # ── Response field types ──────────────────────────────

    async def test_response_id_is_integer(self, client: AsyncClient):
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("t.txt", io.BytesIO(b"t"), "text/plain")},
        )
        data = resp.json()
        assert isinstance(data["id"], int)
        assert isinstance(data["filename"], str)
        assert isinstance(data["content_type"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["created_at"], str)

    async def test_get_by_id_returns_same_as_upload(self, client: AsyncClient):
        """GET /{id} возвращает те же данные, что и upload."""
        user_id = await self._create_user(client)
        upload_resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("same.txt", io.BytesIO(b"same"), "text/plain")},
        )
        upload_data = upload_resp.json()

        get_resp = await client.get(f"/documents/{upload_data['id']}")
        get_data = get_resp.json()

        assert get_data["id"] == upload_data["id"]
        assert get_data["filename"] == upload_data["filename"]
        assert get_data["content_type"] == upload_data["content_type"]
        assert get_data["status"] == upload_data["status"]
