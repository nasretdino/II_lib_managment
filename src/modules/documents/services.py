import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import UploadFile
from src.core import get_logger
from src.core.exceptions import DailyQuotaExhaustedError, DocumentParsingError
from src.modules.rag.services import RagService

from .chunking import split_into_chunks
from .dao import DocumentDAO
from .models import Document
from .parser import extract_text
from .storage import MinioStorage, LocalStorage


logger = get_logger(module="documents", component="service")


class DocumentService:
    def __init__(
        self,
        dao: DocumentDAO,
        storage: MinioStorage | LocalStorage,
        rag_service: RagService | None = None,
    ):
        self.dao = dao
        self.storage = storage
        self.rag = rag_service

    async def _extract_text_from_bytes(
        self,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        suffix = Path(filename).suffix

        def _write_temp_file() -> str:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                return tmp.name

        temp_path = await asyncio.to_thread(_write_temp_file)
        try:
            return await extract_text(temp_path, content_type)
        finally:
            temp_file = Path(temp_path)
            if temp_file.exists():
                os.remove(temp_file)

    async def upload(self, user_id: int, file: UploadFile) -> Document:
        """
        Оркестрация пайплайна загрузки:
        1. Сохранить файл в MinIO
        2. Создать запись в БД (status="pending")
        3. Извлечь текст через parser.extract_text() из временного файла
        4. Разбить на чанки через chunking.split_into_chunks()
        5. Проиндексировать в LightRAG через RagService
        6. Обновить status="ready"
        """
        content = await file.read()
        content_type = file.content_type or "application/octet-stream"
        file_uri = await self.storage.upload_bytes(
            user_id=user_id,
            filename=file.filename,
            content=content,
            content_type=content_type,
        )

        # 2. Создать запись в БД
        doc = await self.dao.add(
            {
                "user_id": user_id,
                "filename": file.filename,
                "content_type": content_type,
                "file_path": file_uri,
                "file_size": len(content),
                "text_content": "",
                "status": "pending",
            },
            flush=True,
        )

        try:
            await self.dao.update_status(doc.id, "processing")

            # 3. Извлечь текст
            text = await self._extract_text_from_bytes(
                filename=doc.filename,
                content=content,
                content_type=doc.content_type,
            )

            # Обновить text_content
            await self.dao.update(
                filters={"id": doc.id},
                values={"text_content": text},
            )

            # 4. Разбить на чанки
            chunks = split_into_chunks(text)
            logger.info(
                "Document {} split into {} chunks", doc.filename, len(chunks)
            )

            # 5. Сохранить количество чанков
            await self.dao.update(
                filters={"id": doc.id},
                values={"chunks_count": len(chunks)},
            )

            # 6. Передать чанки в RagService для индексации через LightRAG
            if self.rag is not None:
                await self.rag.index_document(
                    doc_id=doc.id,
                    chunks=chunks,
                    file_path=file_uri,
                )
                logger.info("Document {} indexed in RAG", doc.filename)

            # 7. Обновить статус
            await self.dao.update_status(doc.id, "ready")
            doc.status = "ready"
            doc.text_content = text
            doc.chunks_count = len(chunks)

        except DailyQuotaExhaustedError:
            logger.error(
                "Daily quota exhausted while processing document {}",
                doc.filename,
            )
            await self.dao.update_status(doc.id, "pending")
            doc.status = "pending"
            raise
        except DocumentParsingError:
            logger.error("Parsing error for document {}", doc.filename)
            await self.dao.update_status(doc.id, "error")
            doc.status = "error"
            raise
        except ValueError as exc:
            logger.error("Parsing error for document {}: {}", doc.filename, exc)
            await self.dao.update_status(doc.id, "error")
            doc.status = "error"
            raise DocumentParsingError(str(exc))
        except Exception:
            logger.exception("Failed to process document {}", doc.filename)
            await self.dao.update_status(doc.id, "error")
            doc.status = "error"
            raise

        return doc

    async def get_by_user(self, user_id: int) -> list[Document]:
        return list(await self.dao.find_by_user(user_id))

    async def get_by_id(self, doc_id: int) -> Document | None:
        return await self.dao.find_one_or_none_by_id(doc_id)

    async def delete(self, doc_id: int) -> bool:
        doc = await self.dao.find_one_or_none_by_id(doc_id)
        if doc is None:
            return False

        # Удалить из RAG-индекса
        if self.rag is not None:
            try:
                await self.rag.delete_document(doc_id, chunks_count=doc.chunks_count or 0)
            except Exception:
                logger.warning("Failed to delete doc {} from RAG index", doc_id)

        # Удалить объект из MinIO, локального хранилища или legacy-файл с диска
        if doc.file_path.startswith("s3://") or doc.file_path.startswith("file://"):
            await self.storage.delete_by_uri(doc.file_path)
        else:
            path = Path(doc.file_path)
            if path.exists():
                os.remove(path)

        deleted_count = await self.dao.delete(filters={"id": doc_id})
        return deleted_count > 0
