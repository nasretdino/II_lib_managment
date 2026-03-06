"""
RagService — обёртка над LightRAG для индексации и поиска.

Использует Gemini для LLM-генерации и эмбеддингов.
PostgreSQL (pgvector) — единое хранилище для графа, векторов, KV и статусов.
LightRAG берёт на себя:
  - Построение графа знаний (сущности + связи)
  - Создание эмбеддингов и их хранение
  - Гибридный поиск: vector similarity + graph traversal
"""

import asyncio
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import EmbeddingFunc

from src.core.config import settings
from .schemas import SearchResult, IndexResult

WORKING_DIR = Path("rag_storage")

# ── Rate Limiter ──────────────────────────────────────────
# Gemini free tier лимиты (скользящее окно 60 с):
#   - embed_content:    100 req/min → ставим 80
#   - generate_content:  20 req/min → ставим 15


def _make_rate_limiter(limit: int, window: float = 60.0):
    """Фабрика: возвращает async-функцию rate limiter со скользящим окном."""
    timestamps: list[float] = []
    lock = asyncio.Lock()

    async def wait() -> None:
        while True:
            async with lock:
                now = time.monotonic()
                while timestamps and timestamps[0] <= now - window:
                    timestamps.pop(0)
                if len(timestamps) < limit:
                    timestamps.append(time.monotonic())
                    return
                sleep_for = timestamps[0] - (now - window) + 0.5

            logger.debug("Rate limit ({}): ожидание {:.1f}с", limit, sleep_for)
            await asyncio.sleep(sleep_for)

    return wait


_wait_embed_rate = _make_rate_limiter(80)
_wait_llm_rate = _make_rate_limiter(15)


def _get_api_key() -> str:
    return settings.llm.gemini_api_key.get_secret_value()


async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """LLM-функция для LightRAG на базе Gemini с rate limiting и retry."""
    import re as _re

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        await _wait_llm_rate()
        try:
            return await gemini_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                keyword_extraction=keyword_extraction,
                api_key=_get_api_key(),
                model_name=settings.llm.model_name,
                **kwargs,
            )
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                wait = min(15.0 * attempt, 90.0)
                match = _re.search(r"retryDelay.*?(\d+)", err_msg)
                if match:
                    wait = float(match.group(1)) + 2.0
                logger.warning(
                    "LLM 429: попытка {}/{}, ожидание {:.0f}с",
                    attempt, max_retries, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise
    # Последняя попытка — без перехвата
    await _wait_llm_rate()
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        keyword_extraction=keyword_extraction,
        api_key=_get_api_key(),
        model_name=settings.llm.model_name,
        **kwargs,
    )


async def _embedding_func_impl(texts: list[str]) -> np.ndarray:
    """Embedding-функция для LightRAG на базе Gemini с rate limiting и retry."""
    import re as _re

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        await _wait_embed_rate()
        try:
            return await gemini_embed.func(
                texts,
                api_key=_get_api_key(),
                model=settings.llm.embedding_model,
                embedding_dim=settings.llm.embedding_dim,
            )
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                wait = min(30.0 * attempt, 120.0)
                match = _re.search(r"retryDelay.*?(\d+)", err_msg)
                if match:
                    wait = float(match.group(1)) + 2.0
                logger.warning(
                    "Embedding 429: попытка {}/{}, ожидание {:.0f}с",
                    attempt, max_retries, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise
    # Последняя попытка — без перехвата
    await _wait_embed_rate()
    return await gemini_embed.func(
        texts,
        api_key=_get_api_key(),
        model=settings.llm.embedding_model,
        embedding_dim=settings.llm.embedding_dim,
    )


def _build_embedding_func() -> EmbeddingFunc:
    """Создать EmbeddingFunc с правильными атрибутами."""
    return EmbeddingFunc(
        embedding_dim=settings.llm.embedding_dim,
        max_token_size=settings.llm.max_token_size,
        func=_embedding_func_impl,
    )


class RagService:
    """
    Тонкая обёртка над LightRAG c PostgreSQL-хранилищем.

    Жизненный цикл:
      1. initialize() — создать экземпляр LightRAG и инициализировать хранилища
      2. index_document() — вставить документ для индексации
      3. search() — поиск по базе знаний
      4. delete_document() — удалить документ из индекса
      5. shutdown() — корректно завершить работу
    """

    def __init__(self) -> None:
        self._rag: LightRAG | None = None

    @property
    def rag(self) -> LightRAG:
        if self._rag is None:
            raise RuntimeError(
                "RagService не инициализирован. Вызовите initialize() перед использованием."
            )
        return self._rag

    async def initialize(self) -> None:
        """Создать экземпляр LightRAG и инициализировать хранилища в PostgreSQL."""
        working_dir = str(WORKING_DIR)
        os.makedirs(working_dir, exist_ok=True)

        self._rag = LightRAG(
            working_dir=working_dir,
            # Изоляция данных
            workspace=settings.llm.workspace,
            # LLM
            llm_model_func=_llm_model_func,
            llm_model_name=settings.llm.model_name,
            # Эмбеддинги
            embedding_func=_build_embedding_func(),
            # Хранилища
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="NetworkXStorage",  # AGE extension недоступна для PG17
            doc_status_storage="PGDocStatusStorage",
            # Чанкинг (LightRAG сам разбивает текст)
            chunk_token_size=settings.llm.chunk_token_size,
            chunk_overlap_token_size=settings.llm.chunk_overlap_token_size,
            # Производительность — Gemini free tier: 20 LLM req/min, 100 embed req/min
            max_parallel_insert=1,
        )

        await self._rag.initialize_storages()
        logger.info(
            "RagService инициализирован (model={}, embedding={}, storage=PostgreSQL, workspace={})",
            settings.llm.model_name,
            settings.llm.embedding_model,
            settings.llm.workspace,
        )

    async def shutdown(self) -> None:
        """Корректно завершить работу LightRAG."""
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None
            logger.info("RagService остановлен")

    async def index_document(
        self,
        doc_id: int,
        chunks: list[str],
        file_path: str | None = None,
    ) -> IndexResult:
        """
        Вставить документ в LightRAG для индексации (граф + векторы).

        Каждый чанк вставляется как отдельный документ с ID вида doc{N}_chunk{M},
        чтобы LightRAG выполнил entity-relationship extraction по каждому.
        """
        if not chunks:
            logger.warning("Документ {} не содержит чанков, пропускаем индексацию", doc_id)
            return IndexResult(doc_id=doc_id, chunks_count=0, status="empty")

        # Добавляем doc_id в каждый чанк для трассировки
        tagged_chunks = [
            f"[doc_id={doc_id}]\n{chunk}" for chunk in chunks
        ]

        ids = [f"doc{doc_id}_chunk{i}" for i in range(len(tagged_chunks))]

        file_paths = (
            [file_path] * len(tagged_chunks) if file_path else None
        )

        await self.rag.ainsert(
            tagged_chunks,
            ids=ids,
            file_paths=file_paths,
        )

        logger.info(
            "Документ {} проиндексирован: {} чанков",
            doc_id,
            len(chunks),
        )
        return IndexResult(doc_id=doc_id, chunks_count=len(chunks))

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        conversation_history: list[dict[str, str]] | None = None,
        only_context: bool = False,
    ) -> SearchResult:
        """
        Поиск через LightRAG.

        Режимы:
          - "naive"  — только vector similarity
          - "local"  — поиск по локальному контексту сущностей
          - "global" — глобальный обзор по всему графу
          - "hybrid" — комбинация local + global (рекомендуемый)
          - "mix"    — knowledge graph + vector retrieval
        """
        param = QueryParam(
            mode=mode,
            only_need_context=only_context,
            conversation_history=conversation_history or [],
        )

        result = await self.rag.aquery(query, param=param)

        # Извлекаем источники из контекста (doc_id-теги)
        sources: list[str] = []
        if isinstance(result, str):
            import re
            found = re.findall(r"\[doc_id=(\d+)\]", result)
            sources = list(set(found))

        return SearchResult(
            context_text=result if isinstance(result, str) else str(result),
            sources=sources,
            mode=mode,
        )

    async def delete_document(self, doc_id: int, chunks_count: int = 0) -> None:
        """
        Удалить все чанки документа из LightRAG.

        LightRAG.adelete_by_doc_id() принимает doc_id — строковый ID,
        присвоенный при ainsert(). Мы использовали формат doc{N}_chunk{M}.
        """
        if chunks_count <= 0:
            logger.warning("Документ {} — chunks_count не задан, пропускаем удаление из RAG", doc_id)
            return

        deleted_any = False
        for i in range(chunks_count):
            chunk_id = f"doc{doc_id}_chunk{i}"
            try:
                await self.rag.adelete_by_doc_id(chunk_id)
                deleted_any = True
            except Exception:
                logger.warning("Ошибка при удалении чанка {} из RAG", chunk_id)
                if deleted_any:
                    break

        logger.info("Документ {} удалён из индекса ({} чанков)", doc_id, chunks_count)


# ── Singleton ─────────────────────────────────────────────
_rag_service: RagService | None = None


def get_rag_service() -> RagService:
    """Получить глобальный экземпляр RagService (singleton)."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service
