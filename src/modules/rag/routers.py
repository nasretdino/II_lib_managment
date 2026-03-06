"""
REST-эндпоинты для поиска по базе знаний (LightRAG).

Предоставляет единственный маршрут POST /rag/search,
который принимает текстовый запрос и режим поиска,
а возвращает контекст из графа знаний.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from .dependencies import get_rag
from .schemas import SearchRequest, SearchResult
from .services import RagService


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/search", response_model=SearchResult)
async def search_knowledge_base(
    body: SearchRequest,
    rag: Annotated[RagService, Depends(get_rag)],
):
    """
    Поиск по базе знаний через LightRAG.

    Режимы:
    - **naive**  — только vector similarity (точный поиск цитат)
    - **local**  — локальный контекст сущностей
    - **global** — глобальный обзор по всему графу
    - **hybrid** — комбинация local + global (рекомендуемый)
    - **mix**    — knowledge graph + vector retrieval
    """
    return await rag.search(
        query=body.query,
        mode=body.mode,
        only_context=body.only_context,
    )
