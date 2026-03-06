import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import DocumentParsingError
from src.modules.documents.dao import DocumentDAO
from src.modules.documents.services import DocumentService
from src.modules.users.dao import UserDAO
from src.modules.users.schemas import UserCreate


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def user(db_session: AsyncSession):
    dao = UserDAO(db_session)
    return await dao.add(UserCreate(name="TestUser"), flush=True)


@pytest.fixture
def doc_service(db_session: AsyncSession) -> DocumentService:
    return DocumentService(DocumentDAO(db_session))


class TestDocumentServiceGetById:
    async def test_get_by_id(self, doc_service: DocumentService, db_session, user):
        """Прямое создание записи через DAO, затем get_by_id через сервис."""
        dao = DocumentDAO(db_session)
        doc = await dao.add(
            {
                "user_id": user.id,
                "filename": "test.txt",
                "content_type": "text/plain",
                "file_path": "/tmp/test.txt",
                "text_content": "hello",
                "status": "ready",
            },
            flush=True,
        )

        found = await doc_service.get_by_id(doc.id)
        assert found is not None
        assert found.filename == "test.txt"

    async def test_get_by_id_not_found(self, doc_service: DocumentService):
        found = await doc_service.get_by_id(9999)
        assert found is None

    async def test_get_by_id_returns_correct_fields(self, doc_service, db_session, user):
        dao = DocumentDAO(db_session)
        doc = await dao.add(
            {
                "user_id": user.id,
                "filename": "report.pdf",
                "content_type": "application/pdf",
                "file_path": "/tmp/report.pdf",
                "text_content": "pdf content",
                "status": "processing",
            },
            flush=True,
        )
        found = await doc_service.get_by_id(doc.id)
        assert found.content_type == "application/pdf"
        assert found.status == "processing"
        assert found.user_id == user.id


class TestDocumentServiceGetByUser:
    async def test_get_by_user(self, doc_service: DocumentService, db_session, user):
        dao = DocumentDAO(db_session)
        for name in ("a.txt", "b.txt"):
            await dao.add(
                {
                    "user_id": user.id,
                    "filename": name,
                    "content_type": "text/plain",
                    "file_path": f"/tmp/{name}",
                    "text_content": "",
                    "status": "ready",
                },
                flush=True,
            )

        docs = await doc_service.get_by_user(user.id)
        assert len(docs) == 2

    async def test_get_by_user_empty(self, doc_service: DocumentService):
        docs = await doc_service.get_by_user(9999)
        assert docs == []

    async def test_get_by_user_isolation(self, doc_service, db_session):
        """Документы одного пользователя не попадают в выборку другого."""
        user_dao = UserDAO(db_session)
        u1 = await user_dao.add(UserCreate(name="User1"), flush=True)
        u2 = await user_dao.add(UserCreate(name="User2"), flush=True)

        doc_dao = DocumentDAO(db_session)
        await doc_dao.add(
            {
                "user_id": u1.id,
                "filename": "u1.txt",
                "content_type": "text/plain",
                "file_path": "/tmp/u1.txt",
                "text_content": "",
                "status": "ready",
            },
            flush=True,
        )
        await doc_dao.add(
            {
                "user_id": u2.id,
                "filename": "u2.txt",
                "content_type": "text/plain",
                "file_path": "/tmp/u2.txt",
                "text_content": "",
                "status": "ready",
            },
            flush=True,
        )

        docs_u1 = await doc_service.get_by_user(u1.id)
        docs_u2 = await doc_service.get_by_user(u2.id)
        assert len(docs_u1) == 1
        assert len(docs_u2) == 1
        assert docs_u1[0].filename == "u1.txt"
        assert docs_u2[0].filename == "u2.txt"


class TestDocumentServiceDelete:
    async def test_delete(self, doc_service: DocumentService, db_session, user, tmp_path):
        file_path = tmp_path / "to_delete.txt"
        file_path.write_text("content")

        dao = DocumentDAO(db_session)
        doc = await dao.add(
            {
                "user_id": user.id,
                "filename": "to_delete.txt",
                "content_type": "text/plain",
                "file_path": str(file_path),
                "text_content": "content",
                "status": "ready",
            },
            flush=True,
        )

        deleted = await doc_service.delete(doc.id)
        assert deleted is True
        assert not file_path.exists()

    async def test_delete_not_found(self, doc_service: DocumentService):
        deleted = await doc_service.delete(9999)
        assert deleted is False

    async def test_delete_file_already_gone(self, doc_service: DocumentService, db_session, user):
        """Удаление документа, файл которого уже отсутствует на диске."""
        dao = DocumentDAO(db_session)
        doc = await dao.add(
            {
                "user_id": user.id,
                "filename": "gone.txt",
                "content_type": "text/plain",
                "file_path": "/tmp/nonexistent_file_12345.txt",
                "text_content": "",
                "status": "ready",
            },
            flush=True,
        )

        deleted = await doc_service.delete(doc.id)
        assert deleted is True

    async def test_delete_removes_from_db(self, doc_service, db_session, user, tmp_path):
        """После удаления документ не находится через get_by_id."""
        file_path = tmp_path / "del.txt"
        file_path.write_text("x")

        dao = DocumentDAO(db_session)
        doc = await dao.add(
            {
                "user_id": user.id,
                "filename": "del.txt",
                "content_type": "text/plain",
                "file_path": str(file_path),
                "text_content": "x",
                "status": "ready",
            },
            flush=True,
        )
        doc_id = doc.id

        await doc_service.delete(doc_id)
        db_session.expunge_all()
        assert await doc_service.get_by_id(doc_id) is None


class TestDocumentServiceUpload:
    async def test_upload_txt(self, doc_service: DocumentService, user):
        """Upload txt-файла создаёт документ со статусом ready."""
        file = UploadFile(
            filename="hello.txt",
            file=io.BytesIO(b"Hello, world!"),
            headers={"content-type": "text/plain"},
        )
        doc = await doc_service.upload(user.id, file)
        assert doc.filename == "hello.txt"
        assert doc.status == "ready"
        assert doc.text_content == "Hello, world!"

    async def test_upload_unsupported_type_sets_error(self, doc_service, user):
        """Upload с неподдерживаемым типом → DocumentParsingError, статус error."""
        file = UploadFile(
            filename="data.bin",
            file=io.BytesIO(b"\x00\x01"),
            headers={"content-type": "application/octet-stream"},
        )
        with pytest.raises(DocumentParsingError):
            await doc_service.upload(user.id, file)

    async def test_upload_empty_txt(self, doc_service, user):
        """Upload пустого txt → ready, text_content пуст."""
        file = UploadFile(
            filename="empty.txt",
            file=io.BytesIO(b""),
            headers={"content-type": "text/plain"},
        )
        doc = await doc_service.upload(user.id, file)
        assert doc.status == "ready"

    async def test_upload_creates_file_on_disk(self, doc_service, user):
        """Upload создаёт файл на диске в uploads/{user_id}/."""
        from pathlib import Path
        file = UploadFile(
            filename="disk.txt",
            file=io.BytesIO(b"disk content"),
            headers={"content-type": "text/plain"},
        )
        doc = await doc_service.upload(user.id, file)
        assert Path(doc.file_path).exists()
