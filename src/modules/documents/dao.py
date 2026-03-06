from collections.abc import Sequence

from src.db import BaseDAO
from .models import Document


class DocumentDAO(BaseDAO[Document]):
    model = Document

    async def find_by_user(self, user_id: int) -> Sequence[Document]:
        """Все документы пользователя."""
        return await self.find_all(filters={"user_id": user_id})

    async def update_status(self, doc_id: int, status: str) -> int:
        """Обновление статуса обработки документа."""
        return await self.update(
            filters={"id": doc_id},
            values={"status": status},
        )
