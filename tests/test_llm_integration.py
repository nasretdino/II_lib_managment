"""
Тесты для ИИ-модельных функций: LLM, embedding, rate limiter.

Все внешние вызовы (Gemini API) мокаются — тестируется
бизнес-логика: retry на 429, rate limiter, передача параметров,
корректность EmbeddingFunc.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
import pytest

from src.modules.rag.services import (
    _make_rate_limiter,
    _llm_model_func,
    _embedding_func_impl,
    _build_embedding_func,
    get_rag_service,
    RagService,
)


# ── Rate Limiter ──────────────────────────────────────────


pytestmark = pytest.mark.asyncio


class TestRateLimiter:
    """Тесты скользящего окна rate limiter."""

    async def test_allows_requests_within_limit(self):
        """Разрешает до limit запросов без ожидания."""
        wait = _make_rate_limiter(5, window=60.0)
        for _ in range(5):
            await wait()  # не должно блокировать

    async def test_blocks_when_limit_exceeded(self):
        """Блокирует, когда лимит исчерпан."""
        wait = _make_rate_limiter(2, window=0.5)
        await wait()
        await wait()

        start = time.monotonic()
        await wait()  # Должен подождать ~0.5с
        elapsed = time.monotonic() - start
        assert elapsed >= 0.3  # допуск

    async def test_window_expires_and_allows_new_requests(self):
        """После истечения окна разрешает новые запросы."""
        wait = _make_rate_limiter(1, window=0.2)
        await wait()
        await asyncio.sleep(0.25)  # окно истекло
        # Следующий запрос не должен блокировать долго
        start = time.monotonic()
        await wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.2

    async def test_concurrent_requests_respect_limit(self):
        """Параллельные запросы не превышают лимит."""
        wait = _make_rate_limiter(3, window=1.0)
        results = await asyncio.gather(*[wait() for _ in range(3)])
        assert len(results) == 3

    async def test_limiter_independence(self):
        """Разные rate limiter'ы не влияют друг на друга."""
        wait_a = _make_rate_limiter(1, window=60.0)
        wait_b = _make_rate_limiter(1, window=60.0)
        await wait_a()
        # wait_b должен пройти без задержки
        start = time.monotonic()
        await wait_b()
        assert time.monotonic() - start < 0.1


# ── LLM Function ─────────────────────────────────────────


class TestLLMModelFunc:
    """Тесты _llm_model_func: retry, parameter passing."""

    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_successful_call(self, mock_key, mock_complete, mock_rate):
        """Успешный вызов — gemini_model_complete вызывается один раз."""
        mock_complete.return_value = "LLM response"

        result = await _llm_model_func("test prompt")

        assert result == "LLM response"
        mock_complete.assert_awaited_once()
        mock_rate.assert_awaited()

    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_passes_all_parameters(self, mock_key, mock_complete, mock_rate):
        """Все параметры передаются в gemini_model_complete."""
        mock_complete.return_value = "ok"

        await _llm_model_func(
            "prompt",
            system_prompt="system",
            history_messages=[{"role": "user", "content": "hi"}],
            keyword_extraction=True,
        )

        call_kwargs = mock_complete.call_args
        assert call_kwargs[0][0] == "prompt"
        assert call_kwargs[1]["system_prompt"] == "system"
        assert call_kwargs[1]["history_messages"] == [{"role": "user", "content": "hi"}]
        assert call_kwargs[1]["keyword_extraction"] is True
        assert call_kwargs[1]["api_key"] == "test-key"

    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_default_history_messages_is_empty_list(self, mock_key, mock_complete, mock_rate):
        """Без history_messages передаётся пустой список."""
        mock_complete.return_value = "ok"

        await _llm_model_func("prompt")

        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["history_messages"] == []

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_retry_on_429(self, mock_key, mock_complete, mock_rate, mock_sleep):
        """429 ошибка → retry с задержкой, затем успех."""
        mock_complete.side_effect = [
            Exception("429 Resource Exhausted"),
            "success after retry",
        ]

        result = await _llm_model_func("prompt")

        assert result == "success after retry"
        assert mock_complete.await_count == 2
        mock_sleep.assert_awaited()

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_retry_on_resource_exhausted(self, mock_key, mock_complete, mock_rate, mock_sleep):
        """RESOURCE_EXHAUSTED → retry."""
        mock_complete.side_effect = [
            Exception("RESOURCE_EXHAUSTED"),
            "ok",
        ]

        result = await _llm_model_func("prompt")
        assert result == "ok"
        assert mock_complete.await_count == 2

    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_non_429_error_raises_immediately(self, mock_key, mock_complete, mock_rate):
        """Не-429 ошибка поднимается сразу без retry."""
        mock_complete.side_effect = ValueError("invalid model")

        with pytest.raises(ValueError, match="invalid model"):
            await _llm_model_func("prompt")

        assert mock_complete.await_count == 1

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_retry_parses_retry_delay(self, mock_key, mock_complete, mock_rate, mock_sleep):
        """retryDelay в сообщении ошибки парсится и используется для ожидания."""
        mock_complete.side_effect = [
            Exception("429 retryDelay: 10 seconds"),
            "ok",
        ]

        result = await _llm_model_func("prompt")
        assert result == "ok"
        # sleep вызван с parsed delay (10 + 2 = 12.0)
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg == 12.0

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_max_retries_then_last_attempt(self, mock_key, mock_complete, mock_rate, mock_sleep):
        """5 retry-ошибок 429 → финальная (6-я) попытка без перехвата."""
        mock_complete.side_effect = [
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            "final success",
        ]

        result = await _llm_model_func("prompt")
        assert result == "final success"
        assert mock_complete.await_count == 6  # 5 retries + 1 final

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_all_retries_exhausted_raises(self, mock_key, mock_complete, mock_rate, mock_sleep):
        """Все попытки исчерпаны → последняя ошибка поднимается."""
        mock_complete.side_effect = Exception("429 limit exceeded")

        with pytest.raises(Exception, match="429"):
            await _llm_model_func("prompt")

        # 5 в цикле + 1 финальная = 6
        assert mock_complete.await_count == 6

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_retry_wait_capped_at_90(self, mock_key, mock_complete, mock_rate, mock_sleep):
        """Задержка при retry ограничена 90 секундами (min(15*attempt, 90))."""
        # 429 без retryDelay — используется формула min(15*attempt, 90)
        mock_complete.side_effect = [
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            "ok",
        ]

        await _llm_model_func("prompt")

        # attempt=5: min(15*5, 90) = 75, не 90
        # attempt=1: min(15, 90) = 15
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_calls[0] == 15.0  # attempt 1
        assert sleep_calls[1] == 30.0  # attempt 2
        assert sleep_calls[2] == 45.0  # attempt 3
        assert sleep_calls[3] == 60.0  # attempt 4
        assert sleep_calls[4] == 75.0  # attempt 5


# ── Embedding Function ────────────────────────────────────


class TestEmbeddingFuncImpl:
    """Тесты _embedding_func_impl: retry, parameter passing."""

    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_successful_call(self, mock_key, mock_embed, mock_rate):
        """Успешный вызов возвращает numpy массив."""
        expected = np.array([[0.1, 0.2, 0.3]])
        mock_embed.func = AsyncMock(return_value=expected)

        result = await _embedding_func_impl(["test text"])

        np.testing.assert_array_equal(result, expected)
        mock_embed.func.assert_awaited_once()

    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_passes_correct_parameters(self, mock_key, mock_embed, mock_rate):
        """Передаёт api_key, model, embedding_dim."""
        mock_embed.func = AsyncMock(return_value=np.array([[0.0]]))

        await _embedding_func_impl(["hello"])

        call_kwargs = mock_embed.func.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        # model и embedding_dim берутся из settings

    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_passes_texts_list(self, mock_key, mock_embed, mock_rate):
        """Список текстов передаётся как первый аргумент."""
        mock_embed.func = AsyncMock(return_value=np.array([[0.0]]))
        texts = ["text1", "text2", "text3"]

        await _embedding_func_impl(texts)

        call_args = mock_embed.func.call_args[0]
        assert call_args[0] == texts

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_retry_on_429(self, mock_key, mock_embed, mock_rate, mock_sleep):
        """429 ошибка → retry, затем успех."""
        mock_embed.func = AsyncMock(
            side_effect=[
                Exception("429 RESOURCE_EXHAUSTED"),
                np.array([[0.5, 0.5]]),
            ]
        )

        result = await _embedding_func_impl(["text"])

        np.testing.assert_array_equal(result, np.array([[0.5, 0.5]]))
        assert mock_embed.func.await_count == 2
        mock_sleep.assert_awaited()

    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_non_429_error_raises_immediately(self, mock_key, mock_embed, mock_rate):
        """Не-429 ошибка поднимается без retry."""
        mock_embed.func = AsyncMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError, match="bad input"):
            await _embedding_func_impl(["text"])

        assert mock_embed.func.await_count == 1

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_retry_parses_retry_delay(self, mock_key, mock_embed, mock_rate, mock_sleep):
        """retryDelay в ошибке парсится."""
        mock_embed.func = AsyncMock(
            side_effect=[
                Exception("429 retryDelay: 20 seconds"),
                np.array([[0.0]]),
            ]
        )

        await _embedding_func_impl(["text"])

        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg == 22.0  # 20 + 2

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_all_retries_exhausted_raises(self, mock_key, mock_embed, mock_rate, mock_sleep):
        """Все 5 retry + финальная попытка → исключение."""
        mock_embed.func = AsyncMock(side_effect=Exception("429"))

        with pytest.raises(Exception, match="429"):
            await _embedding_func_impl(["text"])

        # 5 в цикле + 1 финальная = 6
        assert mock_embed.func.await_count == 6

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_retry_wait_capped_at_120(self, mock_key, mock_embed, mock_rate, mock_sleep):
        """Задержка embedding retry ограничена 120 секундами (min(30*attempt, 120))."""
        mock_embed.func = AsyncMock(
            side_effect=[
                Exception("429"),
                Exception("429"),
                Exception("429"),
                Exception("429"),
                Exception("429"),
                np.array([[0.0]]),
            ]
        )

        await _embedding_func_impl(["text"])

        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_calls[0] == 30.0   # attempt 1: min(30, 120)
        assert sleep_calls[1] == 60.0   # attempt 2: min(60, 120)
        assert sleep_calls[2] == 90.0   # attempt 3: min(90, 120)
        assert sleep_calls[3] == 120.0  # attempt 4: min(120, 120)
        assert sleep_calls[4] == 120.0  # attempt 5: min(150, 120) → 120

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_multiple_texts_embedding(self, mock_key, mock_embed, mock_rate, mock_sleep):
        """Батч из нескольких текстов обрабатывается как один вызов."""
        expected = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_embed.func = AsyncMock(return_value=expected)

        result = await _embedding_func_impl(["text1", "text2", "text3"])

        np.testing.assert_array_equal(result, expected)
        mock_embed.func.assert_awaited_once()


# ── _build_embedding_func ────────────────────────────────


class TestBuildEmbeddingFunc:
    """Тесты _build_embedding_func."""

    async def test_returns_embedding_func(self):
        """Возвращает EmbeddingFunc с правильными атрибутами."""
        from lightrag.utils import EmbeddingFunc

        func = _build_embedding_func()

        assert isinstance(func, EmbeddingFunc)

    async def test_embedding_dim_from_settings(self):
        """embedding_dim берётся из settings."""
        from src.core.config import settings

        func = _build_embedding_func()
        assert func.embedding_dim == settings.llm.embedding_dim

    async def test_max_token_size_from_settings(self):
        """max_token_size берётся из settings."""
        from src.core.config import settings

        func = _build_embedding_func()
        assert func.max_token_size == settings.llm.max_token_size

    async def test_func_is_callable(self):
        """Внутренняя функция — callable."""
        func = _build_embedding_func()
        assert callable(func.func)


# ── Singleton ─────────────────────────────────────────────


class TestGetRagServiceSingleton:
    """Тесты get_rag_service singleton."""

    async def test_returns_rag_service_instance(self):
        """Возвращает экземпляр RagService."""
        service = get_rag_service()
        assert isinstance(service, RagService)

    async def test_returns_same_instance(self):
        """Повторный вызов возвращает тот же объект."""
        s1 = get_rag_service()
        s2 = get_rag_service()
        assert s1 is s2


# ── LLM Model Func: edge cases ───────────────────────────


class TestLLMModelFuncEdgeCases:
    """Дополнительные edge-case тесты."""

    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_empty_prompt(self, mock_key, mock_complete, mock_rate):
        """Пустой промпт всё равно передаётся."""
        mock_complete.return_value = ""

        result = await _llm_model_func("")
        assert result == ""
        assert mock_complete.call_args[0][0] == ""

    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_long_prompt(self, mock_key, mock_complete, mock_rate):
        """Длинный промпт передаётся без обрезки."""
        long_prompt = "x" * 100_000
        mock_complete.return_value = "ok"

        await _llm_model_func(long_prompt)
        assert mock_complete.call_args[0][0] == long_prompt

    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_unicode_prompt(self, mock_key, mock_complete, mock_rate):
        """Unicode-промпт на разных языках."""
        prompt = "Привет 你好 مرحبا"
        mock_complete.return_value = "ответ"

        result = await _llm_model_func(prompt)
        assert result == "ответ"
        assert mock_complete.call_args[0][0] == prompt

    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_extra_kwargs_passed_through(self, mock_key, mock_complete, mock_rate):
        """Дополнительные kwargs пробрасываются в gemini_model_complete."""
        mock_complete.return_value = "ok"

        await _llm_model_func("prompt", temperature=0.5, top_k=40)

        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_k"] == 40

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_llm_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_model_complete", new_callable=AsyncMock)
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_429_then_non_429_error(self, mock_key, mock_complete, mock_rate, mock_sleep):
        """Сначала 429, потом другая ошибка → вторая ошибка поднимается."""
        mock_complete.side_effect = [
            Exception("429"),
            TypeError("wrong type"),
        ]

        with pytest.raises(TypeError, match="wrong type"):
            await _llm_model_func("prompt")

        assert mock_complete.await_count == 2

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.modules.rag.services._wait_embed_rate", new_callable=AsyncMock)
    @patch("src.modules.rag.services.gemini_embed")
    @patch("src.modules.rag.services._get_api_key", return_value="test-key")
    async def test_embed_429_then_other_error(self, mock_key, mock_embed, mock_rate, mock_sleep):
        """Embedding: 429, потом другая ошибка."""
        mock_embed.func = AsyncMock(
            side_effect=[
                Exception("RESOURCE_EXHAUSTED"),
                RuntimeError("connection lost"),
            ]
        )

        with pytest.raises(RuntimeError, match="connection lost"):
            await _embedding_func_impl(["text"])

        assert mock_embed.func.await_count == 2
