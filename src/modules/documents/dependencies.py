from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.modules.rag.dependencies import get_rag
from src.modules.rag.services import RagService
from .dao import DocumentDAO
from .services import DocumentService


async def get_document_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    rag: Annotated[RagService, Depends(get_rag)],
) -> DocumentService:
    return DocumentService(DocumentDAO(db), rag_service=rag)
