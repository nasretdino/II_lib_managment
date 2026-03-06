import asyncio
import os
from pathlib import Path

from fastapi import UploadFile
from loguru import logger

from src.core.exceptions import DocumentParsingError

from .chunking import split_into_chunks
from .dao import DocumentDAO
from .models import Document
from .parser import extract_text

UPLOAD_DIR = Path("uploads")


class DocumentService:
    def __init__(self, dao: DocumentDAO):
        self.dao = dao

    async def upload(self, user_id: int, file: UploadFile) -> Document:
        """
        Оркестрация пайплайна загрузки:
        1. Сохранить файл на диск
        2. Создать запись в БД (status="pending")
        3. Вызвать parser.extract_text() → text_content
        4. Вызвать chunking.split_into_chunks() → список чанков
        5. TODO: передать чанки в RagService для индексации через LightRAG
        6. Обновить status="ready"
        """
        user_dir = UPLOAD_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        file_path = user_dir / file.filename

        # 1. Сохранить файл на диск
        content = await file.read()
        await asyncio.to_thread(file_path.write_bytes, content)

        # 2. Создать запись в БД
        doc = await self.dao.add(
            {
                "user_id": user_id,
                "filename": file.filename,
                "content_type": file.content_type or "application/octet-stream",
                "file_path": str(file_path),
                "text_content": "",
                "status": "pending",
            },
            flush=True,
        )

        try:
            await self.dao.update_status(doc.id, "processing")

            # 3. Извлечь текст
            text = await extract_text(str(file_path), doc.content_type)

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

            # 5. TODO: передать чанки в RagService для индексации
            # await rag_service.index_document(doc.id, chunks)

            # 6. Обновить статус
            await self.dao.update_status(doc.id, "ready")
            doc.status = "ready"
            doc.text_content = text

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

        # Удалить файл с диска
        path = Path(doc.file_path)
        if path.exists():
            os.remove(path)

        deleted_count = await self.dao.delete(filters={"id": doc_id})
        return deleted_count > 0
