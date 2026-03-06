"""Pydantic-схемы для RAG-модуля."""

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Запрос на поиск по базе знаний."""

    query: str = Field(min_length=1, description="Текст поискового запроса")
    mode: Literal["naive", "local", "global", "hybrid", "mix"] = Field(
        default="hybrid",
        description="Режим поиска: naive | local | global | hybrid | mix",
    )
    only_context: bool = Field(
        default=True,
        description="Вернуть только контекст (без генерации ответа LLM)",
    )


class SearchResult(BaseModel):
    """Результат поиска, возвращаемый агентам."""

    context_text: str = Field(description="Собранный контекст из LightRAG")
    sources: list[str] = Field(
        default_factory=list,
        description="Источники (имена документов / chunk IDs)",
    )
    mode: str = Field(description="Режим поиска: naive | local | global | hybrid | mix")


class IndexResult(BaseModel):
    """Результат индексации документа."""

    doc_id: int
    chunks_count: int
    status: str = "indexed"
