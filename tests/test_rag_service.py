"""
Тесты для RAG-модуля.

RagService зависит от LightRAG и Gemini API — в тестах мокаем
внутренний экземпляр LightRAG чтобы изолировать бизнес-логику.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.modules.rag.schemas import SearchResult, SearchRequest, IndexResult
from src.modules.rag.services import RagService


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def rag_service() -> RagService:
    """RagService с замоканным LightRAG-экземпляром."""
    service = RagService()

    mock_rag = MagicMock()
    mock_rag.ainsert = AsyncMock()
    mock_rag.aquery = AsyncMock(return_value="Ответ от LightRAG по контексту [doc_id=1]")
    mock_rag.adelete_by_doc_id = AsyncMock()
    mock_rag.initialize_storages = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()

    service._rag = mock_rag
    return service


# ── Schemas ───────────────────────────────────────────────


class TestSchemas:
    def test_search_result_creation(self):
        result = SearchResult(
            context_text="some context",
            sources=["doc1", "doc2"],
            mode="hybrid",
        )
        assert result.context_text == "some context"
        assert len(result.sources) == 2
        assert result.mode == "hybrid"

    def test_search_result_defaults(self):
        result = SearchResult(context_text="text", mode="local")
        assert result.sources == []

    def test_index_result_creation(self):
        result = IndexResult(doc_id=1, chunks_count=5)
        assert result.doc_id == 1
        assert result.chunks_count == 5
        assert result.status == "indexed"

    def test_index_result_custom_status(self):
        result = IndexResult(doc_id=2, chunks_count=0, status="empty")
        assert result.status == "empty"

    def test_search_request_defaults(self):
        req = SearchRequest(query="test query")
        assert req.mode == "hybrid"
        assert req.only_context is True

    def test_search_request_custom_mode(self):
        req = SearchRequest(query="q", mode="global", only_context=False)
        assert req.mode == "global"
        assert req.only_context is False

    def test_search_request_rejects_empty_query(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_search_request_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", mode="invalid_mode")

    def test_search_request_all_modes_valid(self):
        for mode in ("naive", "local", "global", "hybrid", "mix"):
            req = SearchRequest(query="q", mode=mode)
            assert req.mode == mode


# ── RagService.index_document ─────────────────────────────


class TestIndexDocument:
    async def test_index_document(self, rag_service: RagService):
        """Индексация чанков — ainsert вызывается с тегированными чанками и ids."""
        result = await rag_service.index_document(
            doc_id=1,
            chunks=["chunk1", "chunk2", "chunk3"],
        )

        assert result.doc_id == 1
        assert result.chunks_count == 3
        assert result.status == "indexed"

        rag_service._rag.ainsert.assert_awaited_once()
        call_args = rag_service._rag.ainsert.call_args

        # Чанки должны быть тегированы doc_id
        tagged = call_args[0][0]
        assert len(tagged) == 3
        assert "[doc_id=1]" in tagged[0]
        assert "chunk1" in tagged[0]

        # ids сгенерированы
        assert call_args[1]["ids"] == ["doc1_chunk0", "doc1_chunk1", "doc1_chunk2"]

    async def test_index_document_with_file_path(self, rag_service: RagService):
        result = await rag_service.index_document(
            doc_id=5,
            chunks=["text"],
            file_path="/uploads/5/doc.pdf",
        )

        assert result.chunks_count == 1
        call_kwargs = rag_service._rag.ainsert.call_args[1]
        assert call_kwargs["file_paths"] == ["/uploads/5/doc.pdf"]

    async def test_index_empty_chunks(self, rag_service: RagService):
        """Пустой список чанков — ainsert не вызывается."""
        result = await rag_service.index_document(doc_id=2, chunks=[])

        assert result.chunks_count == 0
        assert result.status == "empty"
        rag_service._rag.ainsert.assert_not_awaited()


# ── RagService.search ─────────────────────────────────────


class TestSearch:
    async def test_search_hybrid(self, rag_service: RagService):
        """Поиск в режиме hybrid."""
        result = await rag_service.search("Что такое Python?")

        assert isinstance(result, SearchResult)
        assert result.mode == "hybrid"
        assert "doc_id=1" not in result.context_text or len(result.sources) > 0

        rag_service._rag.aquery.assert_awaited_once()

    async def test_search_modes(self, rag_service: RagService):
        """Все режимы поиска корректно пробрасываются."""
        for mode in ("naive", "local", "global", "hybrid", "mix"):
            rag_service._rag.aquery.reset_mock()
            rag_service._rag.aquery.return_value = f"result for {mode}"

            result = await rag_service.search("query", mode=mode)
            assert result.mode == mode

    async def test_search_extracts_sources(self, rag_service: RagService):
        """doc_id-теги извлекаются из контекста в sources."""
        rag_service._rag.aquery.return_value = (
            "Текст из [doc_id=3] и [doc_id=7] с информацией [doc_id=3]"
        )

        result = await rag_service.search("query")
        assert "3" in result.sources
        assert "7" in result.sources
        # Дедупликация
        assert result.sources.count("3") == 1

    async def test_search_with_history(self, rag_service: RagService):
        """conversation_history передаётся в QueryParam."""
        history = [{"role": "user", "content": "hello"}]
        await rag_service.search("follow up", conversation_history=history)

        call_args = rag_service._rag.aquery.call_args
        param = call_args[1]["param"]
        assert param.conversation_history == history

    async def test_search_only_context(self, rag_service: RagService):
        """only_context=True передаётся в QueryParam."""
        await rag_service.search("query", only_context=True)

        call_args = rag_service._rag.aquery.call_args
        param = call_args[1]["param"]
        assert param.only_need_context is True


# ── RagService.delete_document ────────────────────────────


class TestDeleteDocument:
    async def test_delete_all_chunks(self, rag_service: RagService):
        """Удаление документа вызывает adelete_by_doc_id для каждого чанка."""
        await rag_service.delete_document(doc_id=5, chunks_count=3)

        assert rag_service._rag.adelete_by_doc_id.await_count == 3
        calls = [c.args[0] for c in rag_service._rag.adelete_by_doc_id.call_args_list]
        assert calls == ["doc5_chunk0", "doc5_chunk1", "doc5_chunk2"]

    async def test_delete_stops_after_last_chunk(self, rag_service: RagService):
        """После удалённых чанков при исключении — останавливается (break)."""
        rag_service._rag.adelete_by_doc_id.side_effect = [
            None, None, Exception("not found"),
        ]
        await rag_service.delete_document(doc_id=5, chunks_count=5)
        # 2 успешных + 1 с ошибкой = 3 вызова, дальше break
        assert rag_service._rag.adelete_by_doc_id.await_count == 3

    async def test_delete_skips_missing_before_any_deleted(self, rag_service: RagService):
        """Если ничего не удалено и исключение — продолжает (continue)."""
        rag_service._rag.adelete_by_doc_id.side_effect = Exception("not found")
        await rag_service.delete_document(doc_id=999, chunks_count=3)
        # Ни один чанк не удалён → continue все 3 итерации
        assert rag_service._rag.adelete_by_doc_id.await_count == 3


# ── RagService lifecycle ──────────────────────────────────


class TestLifecycle:
    async def test_rag_not_initialized_raises(self):
        """Доступ к rag до initialize() выбрасывает RuntimeError."""
        service = RagService()
        with pytest.raises(RuntimeError, match="не инициализирован"):
            _ = service.rag

    async def test_shutdown(self, rag_service: RagService):
        """shutdown() вызывает finalize_storages и сбрасывает _rag."""
        await rag_service.shutdown()

        assert rag_service._rag is None


# ── Router /rag/search ────────────────────────────────────


class TestRagRouter:
    async def test_search_returns_200(self, client):
        """POST /rag/search возвращает 200 с контекстом."""
        resp = await client.post("/rag/search", json={"query": "Что такое Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "context_text" in data
        assert "mode" in data

    async def test_search_custom_mode(self, client):
        """POST /rag/search с mode=global."""
        resp = await client.post(
            "/rag/search",
            json={"query": "обзор", "mode": "global"},
        )
        assert resp.status_code == 200

    async def test_search_empty_query_returns_422(self, client):
        """Пустой query отклоняется валидацией."""
        resp = await client.post("/rag/search", json={"query": ""})
        assert resp.status_code == 422

    async def test_search_invalid_mode_returns_422(self, client):
        """Невалидный mode отклоняется."""
        resp = await client.post(
            "/rag/search",
            json={"query": "test", "mode": "invalid"},
        )
        assert resp.status_code == 422

    async def test_search_response_structure(self, client):
        """Проверка полной структуры ответа."""
        resp = await client.post("/rag/search", json={"query": "test"})
        data = resp.json()
        assert isinstance(data["context_text"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["mode"], str)

    async def test_search_missing_body_returns_422(self, client):
        """Запрос без тела отклоняется."""
        resp = await client.post("/rag/search")
        assert resp.status_code == 422

    async def test_search_default_mode_is_hybrid(self, client):
        """Без указания mode по умолчанию hybrid."""
        resp = await client.post("/rag/search", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["mode"] == "hybrid"

    async def test_search_all_valid_modes(self, client):
        """Все 5 режимов принимаются."""
        for mode in ("naive", "local", "global", "hybrid", "mix"):
            resp = await client.post(
                "/rag/search",
                json={"query": "q", "mode": mode},
            )
            assert resp.status_code == 200
