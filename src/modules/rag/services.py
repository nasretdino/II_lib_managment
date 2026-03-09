"""
RagService — обёртка над LightRAG для индексации и поиска.

Провайдер-агностичная реализация: конкретный LLM-провайдер
(Gemini, OpenAI, etc.) подключается через LLMProvider.

Storage-модель:
    - PostgreSQL (pgvector): KV, векторы, статусы документов
    - Neo4j: граф сущностей и связей (Graph RAG)
"""

import asyncio
import os
import random
import re
import time
from pathlib import Path

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from src.core import get_logger
from src.core.config import settings
from src.core.exceptions import DailyQuotaExhaustedError
from .llm_provider import LLMProvider, create_provider
from .schemas import SearchResult, IndexResult


logger = get_logger(module="rag", component="service")

WORKING_DIR = Path("rag_storage")


# ── Rate Limiter ──────────────────────────────────────────


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


class RagService:
    """
    Обёртка над LightRAG, провайдер-агностичная.

    Жизненный цикл:
      1. initialize() — создать экземпляр LightRAG и инициализировать хранилища
      2. index_document() — вставить документ для индексации
      3. search() — поиск по базе знаний
      4. delete_document() — удалить документ из индекса
      5. shutdown() — корректно завершить работу
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._rag: LightRAG | None = None
        self._provider = provider

        # Rate limiters из конфига
        self._wait_llm_rate = _make_rate_limiter(
            settings.llm.llm_rate_limit,
            settings.llm.rate_limit_window,
        )
        self._wait_embed_rate = _make_rate_limiter(
            settings.llm.embed_rate_limit,
            settings.llm.rate_limit_window,
        )

    @property
    def rag(self) -> LightRAG:
        if self._rag is None:
            raise RuntimeError(
                "RagService не инициализирован. Вызовите initialize() перед использованием."
            )
        return self._rag

    # ── helpers ────────────────────────────────────────────

    @staticmethod
    def _is_daily_quota(err_msg: str) -> bool:
        """Detect if the 429 is a *daily* (free-tier) quota, not per-minute."""
        return "FreeTier" in err_msg or "PerDay" in err_msg

    # ── LLM wrapper (rate limit + retry) ──────────────────

    async def _llm_func(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        """Rate-limited + retry обёртка над provider.complete()."""
        max_retries = settings.llm.max_retries
        for attempt in range(1, max_retries + 1):
            await self._wait_llm_rate()
            try:
                return await self._provider.complete(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    keyword_extraction=keyword_extraction,
                    **kwargs,
                )
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    # Daily free-tier quota — retrying won't help
                    if self._is_daily_quota(err_msg):
                        provider_name = settings.llm.provider
                        logger.error(
                            "Дневной лимит бесплатного тарифа {} исчерпан. "
                            "Повторные попытки бесполезны. Обновите тариф или "
                            "дождитесь сброса квоты.",
                            provider_name,
                        )
                        raise DailyQuotaExhaustedError(
                            f"Дневной лимит запросов {provider_name} (free tier) исчерпан. "
                            "Перейдите на платный тариф или дождитесь сброса квоты."
                        ) from e

                    # Per-minute rate limit — exponential backoff + jitter
                    wait = min(
                        settings.llm.llm_retry_base_delay * (2 ** (attempt - 1)),
                        settings.llm.llm_retry_max_delay,
                    )
                    match = re.search(r"retryDelay.*?(\d+)", err_msg)
                    if match:
                        wait = float(match.group(1)) + 2.0
                    wait += random.uniform(0, 5)
                    logger.warning(
                        "LLM 429: попытка {}/{}, ожидание {:.0f}с",
                        attempt, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        # Последняя попытка — без перехвата
        await self._wait_llm_rate()
        return await self._provider.complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            **kwargs,
        )

    async def _embed_func(self, texts: list[str]) -> np.ndarray:
        """Rate-limited + retry обёртка над provider.embed()."""
        max_retries = settings.llm.max_retries
        for attempt in range(1, max_retries + 1):
            await self._wait_embed_rate()
            try:
                return await self._provider.embed(texts)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    if self._is_daily_quota(err_msg):
                        provider_name = settings.llm.provider
                        logger.error(
                            "Дневной лимит бесплатного тарифа {} (embed) исчерпан.",
                            provider_name,
                        )
                        raise DailyQuotaExhaustedError(
                            f"Дневной лимит запросов {provider_name} embedding (free tier) исчерпан."
                        ) from e

                    wait = min(
                        settings.llm.embed_retry_base_delay * (2 ** (attempt - 1)),
                        settings.llm.embed_retry_max_delay,
                    )
                    match = re.search(r"retryDelay.*?(\d+)", err_msg)
                    if match:
                        wait = float(match.group(1)) + 2.0
                    wait += random.uniform(0, 5)
                    logger.warning(
                        "Embedding 429: попытка {}/{}, ожидание {:.0f}с",
                        attempt, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        # Последняя попытка — без перехвата
        await self._wait_embed_rate()
        return await self._provider.embed(texts)

    def _build_embedding_func(self) -> EmbeddingFunc:
        """Создать EmbeddingFunc для LightRAG."""
        return EmbeddingFunc(
            embedding_dim=self._provider.embedding_dim,
            max_token_size=self._provider.max_token_size,
            func=self._embed_func,
        )

    def _sync_lightrag_storage_env(self) -> None:
        """Передать LightRAG storage-настройки через env-переменные."""
        os.environ["POSTGRES_HOST"] = settings.db.host
        os.environ["POSTGRES_PORT"] = str(settings.db.port)
        os.environ["POSTGRES_USER"] = settings.db.user
        os.environ["POSTGRES_PASSWORD"] = settings.db.password.get_secret_value()
        os.environ["POSTGRES_DATABASE"] = settings.db.name
        os.environ["POSTGRES_WORKSPACE"] = settings.llm.workspace

        os.environ["NEO4J_URI"] = settings.neo4j.uri
        os.environ["NEO4J_USERNAME"] = settings.neo4j.user
        os.environ["NEO4J_PASSWORD"] = settings.neo4j.password.get_secret_value()
        os.environ["NEO4J_DATABASE"] = settings.neo4j.database
        os.environ["NEO4J_WORKSPACE"] = settings.neo4j.workspace

    # ── Lifecycle ─────────────────────────────────────────

    async def initialize(self) -> None:
        """Создать экземпляр LightRAG и инициализировать хранилища."""
        self._sync_lightrag_storage_env()

        working_dir = str(WORKING_DIR)
        os.makedirs(working_dir, exist_ok=True)

        self._rag = LightRAG(
            working_dir=working_dir,
            workspace=settings.llm.workspace,
            llm_model_func=self._llm_func,
            llm_model_name=self._provider.model_name,
            embedding_func=self._build_embedding_func(),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage=settings.rag.graph_storage,
            doc_status_storage="PGDocStatusStorage",
            chunk_token_size=settings.llm.chunk_token_size,
            chunk_overlap_token_size=settings.llm.chunk_overlap_token_size,
            max_parallel_insert=1,
        )

        await self._rag.initialize_storages()
        logger.info(
            "RagService инициализирован (provider={}, model={}, embedding={}, workspace={}, graph_storage={})",
            settings.llm.provider,
            self._provider.model_name,
            self._provider.embedding_model,
            settings.llm.workspace,
            settings.rag.graph_storage,
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
        provider = create_provider(settings.llm)
        _rag_service = RagService(provider)
    return _rag_service
