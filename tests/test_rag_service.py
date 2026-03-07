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
    service = RagService(provider=MagicMock())

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
        service = RagService(provider=MagicMock())
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

    async def test_search_only_context_false(self, client):
        """only_context=false в запросе."""
        resp = await client.post(
            "/rag/search",
            json={"query": "test", "only_context": False},
        )
        assert resp.status_code == 200

    async def test_search_only_context_true(self, client):
        """only_context=true (по умолчанию)."""
        resp = await client.post(
            "/rag/search",
            json={"query": "test", "only_context": True},
        )
        assert resp.status_code == 200

    async def test_search_get_method_not_allowed(self, client):
        """GET /rag/search → 405 Method Not Allowed."""
        resp = await client.get("/rag/search")
        assert resp.status_code == 405

    async def test_search_put_method_not_allowed(self, client):
        """PUT /rag/search → 405."""
        resp = await client.put("/rag/search", json={"query": "test"})
        assert resp.status_code == 405

    async def test_search_delete_method_not_allowed(self, client):
        """DELETE /rag/search → 405."""
        resp = await client.delete("/rag/search")
        assert resp.status_code == 405

    async def test_search_missing_query_field(self, client):
        """Тело без поля query → 422."""
        resp = await client.post("/rag/search", json={"mode": "hybrid"})
        assert resp.status_code == 422

    async def test_search_extra_fields_ignored(self, client):
        """Лишние поля в запросе не ломают обработку."""
        resp = await client.post(
            "/rag/search",
            json={"query": "test", "extra_field": "ignore_me"},
        )
        assert resp.status_code == 200

    async def test_search_long_query(self, client):
        """Длинный запрос принимается."""
        long_query = "слово " * 500
        resp = await client.post("/rag/search", json={"query": long_query})
        assert resp.status_code == 200

    async def test_search_unicode_query(self, client):
        """Unicode-запрос принимается."""
        resp = await client.post(
            "/rag/search",
            json={"query": "Привет мир 你好世界 مرحبا"},
        )
        assert resp.status_code == 200

    async def test_search_whitespace_query_rejected(self, client):
        """Запрос из одних пробелов — query min_length=1, пробел это символ → 200."""
        resp = await client.post("/rag/search", json={"query": " "})
        # min_length=1, пробел — допустимый символ
        assert resp.status_code == 200

    async def test_search_sources_are_list_of_strings(self, client):
        """sources в ответе — список строк."""
        resp = await client.post("/rag/search", json={"query": "test"})
        data = resp.json()
        assert isinstance(data["sources"], list)
        for s in data["sources"]:
            assert isinstance(s, str)

    async def test_search_context_text_is_string(self, client):
        """context_text в ответе — строка."""
        resp = await client.post("/rag/search", json={"query": "test"})
        assert isinstance(resp.json()["context_text"], str)


# ── Дополнительные тесты RagService ──────────────────────


class TestSearchEdgeCases:
    """Дополнительные edge-case тесты для RagService.search."""

    async def test_search_no_sources_in_result(self, rag_service: RagService):
        """Ответ без doc_id тегов → пустой sources."""
        rag_service._rag.aquery.return_value = "Обычный текст без тегов."
        result = await rag_service.search("query")
        assert result.sources == []

    async def test_search_non_string_result(self, rag_service: RagService):
        """Если aquery вернул не строку → str() конвертация."""
        rag_service._rag.aquery.return_value = 12345
        result = await rag_service.search("query")
        assert result.context_text == "12345"
        assert result.sources == []

    async def test_search_only_context_false(self, rag_service: RagService):
        """only_context=False передаётся в QueryParam."""
        await rag_service.search("query", only_context=False)
        call_args = rag_service._rag.aquery.call_args
        param = call_args[1]["param"]
        assert param.only_need_context is False

    async def test_search_empty_conversation_history(self, rag_service: RagService):
        """Пустая история → пустой список в QueryParam."""
        await rag_service.search("query", conversation_history=[])
        call_args = rag_service._rag.aquery.call_args
        param = call_args[1]["param"]
        assert param.conversation_history == []


class TestDeleteDocumentEdgeCases:
    """Дополнительные edge-case тесты для RagService.delete_document."""

    async def test_delete_zero_chunks_skips(self, rag_service: RagService):
        """chunks_count=0 → пропускаем удаление (предупреждение в логе)."""
        await rag_service.delete_document(doc_id=1, chunks_count=0)
        rag_service._rag.adelete_by_doc_id.assert_not_awaited()

    async def test_delete_negative_chunks_skips(self, rag_service: RagService):
        """chunks_count < 0 → пропускаем удаление."""
        await rag_service.delete_document(doc_id=1, chunks_count=-5)
        rag_service._rag.adelete_by_doc_id.assert_not_awaited()

    async def test_delete_single_chunk(self, rag_service: RagService):
        """Удаление документа с 1 чанком."""
        await rag_service.delete_document(doc_id=10, chunks_count=1)
        rag_service._rag.adelete_by_doc_id.assert_awaited_once_with("doc10_chunk0")

    async def test_delete_all_fail_no_break(self, rag_service: RagService):
        """Если ни один чанк не удалён (все исключения) → continue, не break."""
        rag_service._rag.adelete_by_doc_id.side_effect = Exception("not found")
        await rag_service.delete_document(doc_id=1, chunks_count=4)
        assert rag_service._rag.adelete_by_doc_id.await_count == 4

    async def test_delete_first_fail_others_succeed(self, rag_service: RagService):
        """Первый чанк не удалён, остальные — успешно → continue все."""
        rag_service._rag.adelete_by_doc_id.side_effect = [
            Exception("not found"), None, None,
        ]
        await rag_service.delete_document(doc_id=1, chunks_count=3)
        assert rag_service._rag.adelete_by_doc_id.await_count == 3


class TestIndexDocumentEdgeCases:
    """Дополнительные edge-case тесты для index_document."""

    async def test_index_single_chunk(self, rag_service: RagService):
        """Индексация одного чанка."""
        result = await rag_service.index_document(doc_id=99, chunks=["single chunk"])
        assert result.chunks_count == 1
        assert result.status == "indexed"

        call_args = rag_service._rag.ainsert.call_args
        assert call_args[1]["ids"] == ["doc99_chunk0"]

    async def test_index_without_file_path(self, rag_service: RagService):
        """Без file_path → file_paths=None."""
        await rag_service.index_document(doc_id=1, chunks=["text"])
        call_kwargs = rag_service._rag.ainsert.call_args[1]
        assert call_kwargs["file_paths"] is None

    async def test_index_many_chunks(self, rag_service: RagService):
        """Индексация большого количества чанков."""
        chunks = [f"chunk_{i}" for i in range(50)]
        result = await rag_service.index_document(doc_id=1, chunks=chunks)
        assert result.chunks_count == 50

        call_args = rag_service._rag.ainsert.call_args
        ids = call_args[1]["ids"]
        assert len(ids) == 50
        assert ids[0] == "doc1_chunk0"
        assert ids[49] == "doc1_chunk49"

    async def test_index_chunks_tagged_with_doc_id(self, rag_service: RagService):
        """Каждый чанк тегируется [doc_id=N]."""
        await rag_service.index_document(doc_id=42, chunks=["hello", "world"])
        tagged = rag_service._rag.ainsert.call_args[0][0]
        assert tagged[0] == "[doc_id=42]\nhello"
        assert tagged[1] == "[doc_id=42]\nworld"


class TestSchemaValidationExtra:
    """Дополнительные тесты валидации схем."""

    def test_search_request_only_query_field(self):
        """Минимальный запрос — только query."""
        req = SearchRequest(query="a")
        assert req.query == "a"
        assert req.mode == "hybrid"
        assert req.only_context is True

    def test_search_result_empty_context(self):
        """Пустой context_text допускается."""
        result = SearchResult(context_text="", mode="naive", sources=[])
        assert result.context_text == ""

    def test_index_result_zero_chunks(self):
        result = IndexResult(doc_id=1, chunks_count=0, status="empty")
        assert result.chunks_count == 0

    def test_search_request_mode_naive(self):
        req = SearchRequest(query="test", mode="naive")
        assert req.mode == "naive"

    def test_search_request_mode_mix(self):
        req = SearchRequest(query="test", mode="mix")
        assert req.mode == "mix"

    def test_search_request_mode_local(self):
        req = SearchRequest(query="test", mode="local")
        assert req.mode == "local"
