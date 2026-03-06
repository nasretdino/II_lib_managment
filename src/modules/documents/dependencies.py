from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from .dao import DocumentDAO
from .services import DocumentService


async def get_document_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentService:
    return DocumentService(DocumentDAO(db))
