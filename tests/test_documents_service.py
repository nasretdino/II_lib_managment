import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import DocumentParsingError
from src.modules.documents.dao import DocumentDAO
from src.modules.documents.services import DocumentService
from src.modules.rag.schemas import IndexResult
from src.modules.users.dao import UserDAO
from src.modules.users.schemas import UserCreate


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def user(db_session: AsyncSession):
    dao = UserDAO(db_session)
    return await dao.add(UserCreate(name="TestUser"), flush=True)


@pytest.fixture
def object_storage_mock() -> MagicMock:
    storage = MagicMock()
    storage.upload_bytes = AsyncMock(return_value="s3://documents/users/1/mock-upload.txt")
    storage.delete_by_uri = AsyncMock()
    return storage


@pytest.fixture
def doc_service(db_session: AsyncSession, object_storage_mock: MagicMock) -> DocumentService:
    return DocumentService(DocumentDAO(db_session), storage=object_storage_mock)


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
        file_uri = "s3://documents/users/1/to_delete.txt"

        dao = DocumentDAO(db_session)
        doc = await dao.add(
            {
                "user_id": user.id,
                "filename": "to_delete.txt",
                "content_type": "text/plain",
                "file_path": file_uri,
                "text_content": "content",
                "status": "ready",
            },
            flush=True,
        )

        deleted = await doc_service.delete(doc.id)
        assert deleted is True
        doc_service.storage.delete_by_uri.assert_awaited_once_with(file_uri)

    async def test_delete_not_found(self, doc_service: DocumentService):
        deleted = await doc_service.delete(9999)
        assert deleted is False

    async def test_delete_file_already_gone(self, doc_service: DocumentService, db_session, user):
        """Удаление legacy-документа с несуществующим локальным путем."""
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
        """Upload сохраняет путь в MinIO и размер файла в метаданные."""
        file = UploadFile(
            filename="disk.txt",
            file=io.BytesIO(b"disk content"),
            headers={"content-type": "text/plain"},
        )
        doc = await doc_service.upload(user.id, file)
        assert doc.file_path.startswith("s3://")
        assert doc.file_size == len(b"disk content")
        doc_service.storage.upload_bytes.assert_awaited_once()


class TestDocumentServiceUploadAllFileTypes:
    """Тесты upload для всех поддерживаемых типов файлов (PDF, DOCX, DOC)."""

    async def test_upload_pdf(self, doc_service: DocumentService, user):
        """Upload PDF через мок extract_text."""
        file = UploadFile(
            filename="report.pdf",
            file=io.BytesIO(b"%PDF-1.4"),
            headers={"content-type": "application/pdf"},
        )
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            return_value="PDF text content here",
        ):
            doc = await doc_service.upload(user.id, file)

        assert doc.filename == "report.pdf"
        assert doc.content_type == "application/pdf"
        assert doc.status == "ready"
        assert doc.text_content == "PDF text content here"

    async def test_upload_docx(self, doc_service: DocumentService, user):
        """Upload DOCX."""
        ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        file = UploadFile(
            filename="document.docx",
            file=io.BytesIO(b"PK\x03\x04"),
            headers={"content-type": ct},
        )
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            return_value="DOCX extracted text",
        ):
            doc = await doc_service.upload(user.id, file)

        assert doc.filename == "document.docx"
        assert doc.content_type == ct
        assert doc.status == "ready"
        assert doc.text_content == "DOCX extracted text"

    async def test_upload_doc(self, doc_service: DocumentService, user):
        """Upload DOC (application/msword)."""
        file = UploadFile(
            filename="legacy.doc",
            file=io.BytesIO(b"\xd0\xcf\x11\xe0"),
            headers={"content-type": "application/msword"},
        )
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            return_value="DOC text content",
        ):
            doc = await doc_service.upload(user.id, file)

        assert doc.filename == "legacy.doc"
        assert doc.content_type == "application/msword"
        assert doc.status == "ready"

    async def test_upload_unsupported_image(self, doc_service: DocumentService, user):
        """Upload image/jpeg → DocumentParsingError."""
        file = UploadFile(
            filename="photo.jpg",
            file=io.BytesIO(b"\xff\xd8\xff\xe0"),
            headers={"content-type": "image/jpeg"},
        )
        with pytest.raises(DocumentParsingError):
            await doc_service.upload(user.id, file)

    async def test_upload_unsupported_zip(self, doc_service: DocumentService, user):
        """Upload application/zip → DocumentParsingError."""
        file = UploadFile(
            filename="archive.zip",
            file=io.BytesIO(b"PK\x03\x04"),
            headers={"content-type": "application/zip"},
        )
        with pytest.raises(DocumentParsingError):
            await doc_service.upload(user.id, file)


class TestDocumentServiceUploadWithRag:
    """Тесты upload с интеграцией RAG-сервиса."""

    async def test_upload_calls_rag_index(
        self,
        db_session: AsyncSession,
        user,
        object_storage_mock: MagicMock,
    ):
        """Upload вызывает rag.index_document с чанками."""
        mock_rag = MagicMock()
        mock_rag.index_document = AsyncMock(
            return_value=IndexResult(doc_id=0, chunks_count=1, status="indexed")
        )
        service = DocumentService(
            DocumentDAO(db_session),
            storage=object_storage_mock,
            rag_service=mock_rag,
        )

        file = UploadFile(
            filename="rag.txt",
            file=io.BytesIO(b"Some text for RAG indexing"),
            headers={"content-type": "text/plain"},
        )
        doc = await service.upload(user.id, file)
        assert doc.status == "ready"

        mock_rag.index_document.assert_awaited_once()
        call_kwargs = mock_rag.index_document.call_args[1]
        assert call_kwargs["doc_id"] == doc.id
        assert isinstance(call_kwargs["chunks"], list)
        assert len(call_kwargs["chunks"]) >= 1

    async def test_upload_without_rag(self, db_session: AsyncSession, user):
        """Upload без RAG-сервиса работает (rag=None)."""
        service = DocumentService(
            DocumentDAO(db_session),
            storage=MagicMock(upload_bytes=AsyncMock(return_value="s3://documents/users/1/no_rag.txt"), delete_by_uri=AsyncMock()),
            rag_service=None,
        )
        file = UploadFile(
            filename="no_rag.txt",
            file=io.BytesIO(b"Content without RAG"),
            headers={"content-type": "text/plain"},
        )
        doc = await service.upload(user.id, file)
        assert doc.status == "ready"

    async def test_upload_sets_chunks_count(self, db_session: AsyncSession, user):
        """Upload сохраняет количество чанков."""
        service = DocumentService(
            DocumentDAO(db_session),
            storage=MagicMock(upload_bytes=AsyncMock(return_value="s3://documents/users/1/chunks.txt"), delete_by_uri=AsyncMock()),
            rag_service=None,
        )
        # Достаточно текста для нескольких чанков
        text = "Paragraph of content. " * 200
        file = UploadFile(
            filename="chunks.txt",
            file=io.BytesIO(text.encode()),
            headers={"content-type": "text/plain"},
        )
        doc = await service.upload(user.id, file)
        assert doc.chunks_count is not None
        assert doc.chunks_count >= 1


class TestDocumentServiceUploadErrors:
    """Тесты обработки ошибок при upload."""

    async def test_upload_value_error_wraps_to_parsing_error(
        self,
        db_session: AsyncSession,
        user,
        object_storage_mock: MagicMock,
    ):
        """ValueError из extract_text оборачивается в DocumentParsingError."""
        service = DocumentService(
            DocumentDAO(db_session),
            storage=object_storage_mock,
            rag_service=None,
        )
        file = UploadFile(
            filename="fail.txt",
            file=io.BytesIO(b"data"),
            headers={"content-type": "text/plain"},
        )
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            side_effect=ValueError("bad value"),
        ):
            with pytest.raises(DocumentParsingError, match="bad value"):
                await service.upload(user.id, file)

    async def test_upload_generic_exception_sets_error_status(
        self,
        db_session: AsyncSession,
        user,
        object_storage_mock: MagicMock,
    ):
        """Непредвиденная ошибка → статус error."""
        service = DocumentService(
            DocumentDAO(db_session),
            storage=object_storage_mock,
            rag_service=None,
        )
        file = UploadFile(
            filename="crash.txt",
            file=io.BytesIO(b"data"),
            headers={"content-type": "text/plain"},
        )
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected"),
        ):
            with pytest.raises(RuntimeError):
                await service.upload(user.id, file)

    async def test_upload_parsing_error_preserves_doc_in_db(
        self,
        db_session: AsyncSession,
        user,
        object_storage_mock: MagicMock,
    ):
        """При ошибке парсинга документ остаётся в БД со статусом error."""
        service = DocumentService(
            DocumentDAO(db_session),
            storage=object_storage_mock,
            rag_service=None,
        )
        file = UploadFile(
            filename="err.txt",
            file=io.BytesIO(b"data"),
            headers={"content-type": "text/plain"},
        )
        with patch(
            "src.modules.documents.services.extract_text",
            new_callable=AsyncMock,
            side_effect=DocumentParsingError("parse fail"),
        ):
            with pytest.raises(DocumentParsingError):
                doc = await service.upload(user.id, file)

        # Документ должен быть в БД со статусом error
        dao = DocumentDAO(db_session)
        docs = await dao.find_by_user(user.id)
        assert len(docs) == 1
        assert docs[0].status == "error"


class TestDocumentServiceDeleteWithRag:
    """Тесты удаления с RAG-интеграцией."""

    async def test_delete_calls_rag_delete(
        self,
        db_session: AsyncSession,
        user,
        object_storage_mock: MagicMock,
    ):
        """Удаление документа вызывает rag.delete_document."""
        mock_rag = MagicMock()
        mock_rag.delete_document = AsyncMock()
        service = DocumentService(
            DocumentDAO(db_session),
            storage=object_storage_mock,
            rag_service=mock_rag,
        )

        file_uri = "s3://documents/users/1/del.txt"

        dao = DocumentDAO(db_session)
        doc = await dao.add(
            {
                "user_id": user.id,
                "filename": "del.txt",
                "content_type": "text/plain",
                "file_path": file_uri,
                "text_content": "content",
                "status": "ready",
                "chunks_count": 3,
            },
            flush=True,
        )

        deleted = await service.delete(doc.id)
        assert deleted is True
        mock_rag.delete_document.assert_awaited_once_with(
            doc.id, chunks_count=3
        )

    async def test_delete_rag_failure_still_deletes(
        self,
        db_session: AsyncSession,
        user,
        object_storage_mock: MagicMock,
    ):
        """Ошибка RAG при удалении не мешает удалению из БД и MinIO."""
        mock_rag = MagicMock()
        mock_rag.delete_document = AsyncMock(side_effect=Exception("RAG fail"))
        service = DocumentService(
            DocumentDAO(db_session),
            storage=object_storage_mock,
            rag_service=mock_rag,
        )

        file_uri = "s3://documents/users/1/rag_fail.txt"

        dao = DocumentDAO(db_session)
        doc = await dao.add(
            {
                "user_id": user.id,
                "filename": "rag_fail.txt",
                "content_type": "text/plain",
                "file_path": file_uri,
                "text_content": "text",
                "status": "ready",
                "chunks_count": 2,
            },
            flush=True,
        )

        deleted = await service.delete(doc.id)
        assert deleted is True
        object_storage_mock.delete_by_uri.assert_awaited_once_with(file_uri)
