from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, UploadFile, status

from src.core.exceptions import NotFoundError

from .dependencies import get_document_service
from .schemas import DocumentRead
from .services import DocumentService


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    user_id: Annotated[int, Query(gt=0)],
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    return await service.upload(user_id, file)


@router.get("/", response_model=list[DocumentRead])
async def get_documents(
    user_id: Annotated[int, Query(gt=0)],
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    return await service.get_by_user(user_id)


@router.get("/{doc_id}", response_model=DocumentRead)
async def get_document(
    doc_id: Annotated[int, Path(gt=0)],
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    doc = await service.get_by_id(doc_id)
    if doc is None:
        raise NotFoundError("Document not found")
    return doc


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: Annotated[int, Path(gt=0)],
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    deleted = await service.delete(doc_id)
    if not deleted:
        raise NotFoundError("Document not found")
