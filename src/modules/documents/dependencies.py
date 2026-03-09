from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.modules.rag.dependencies import get_rag
from src.modules.rag.services import RagService
from .dao import DocumentDAO
from .services import DocumentService
from .storage import MinioStorage, LocalStorage, get_object_storage


async def get_document_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    rag: Annotated[RagService, Depends(get_rag)],
    storage: Annotated[MinioStorage | LocalStorage, Depends(get_object_storage)],
) -> DocumentService:
    return DocumentService(DocumentDAO(db), storage=storage, rag_service=rag)
