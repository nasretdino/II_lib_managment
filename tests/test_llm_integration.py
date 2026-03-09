"""
Тесты для LLM-провайдера, RagService wrappers и rate limiter.

Все внешние вызовы (Gemini API) мокаются — тестируется
бизнес-логика: provider abstraction, retry на 429, rate limiter,
передача параметров, корректность EmbeddingFunc.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock

import numpy as np
import pytest
from pydantic import SecretStr

from src.core.config import LLMSettings
from src.modules.rag.llm_provider import (
    LLMProvider,
    GeminiProvider,
    OllamaProvider,
    create_provider,
)
from src.modules.rag.services import (
    _make_rate_limiter,
    get_rag_service,
    RagService,
)


# ── Helpers ───────────────────────────────────────────────


def _make_llm_settings(**overrides) -> LLMSettings:
    """Создать LLMSettings с тестовыми значениями."""
    defaults = {
        "provider": "gemini",
        "api_key": SecretStr("test-api-key"),
        "ollama_host": "http://localhost:11434",
        "model_name": "gemini-2.5-flash",
        "embedding_model": "gemini-embedding-001",
        "embedding_dim": 768,
        "max_token_size": 8192,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)


def _make_mock_provider() -> MagicMock:
    """MagicMock с атрибутами LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.model_name = "test-model"
    provider.embedding_model = "test-embed-model"
    provider.embedding_dim = 768
    provider.max_token_size = 8192
    provider.complete = AsyncMock(return_value="LLM response")
    provider.embed = AsyncMock(return_value=np.array([[0.1, 0.2, 0.3]]))
    return provider


def _make_service(provider: MagicMock | None = None) -> RagService:
    """Создать RagService с мок-провайдером и отключённым rate limiter."""
    svc = RagService(provider or _make_mock_provider())
    svc._wait_llm_rate = AsyncMock()
    svc._wait_embed_rate = AsyncMock()
    return svc


# ── Rate Limiter ──────────────────────────────────────────


@pytest.mark.asyncio
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


# ── LLM Provider ─────────────────────────────────────────


@pytest.mark.asyncio
class TestGeminiProvider:
    """Тесты GeminiProvider: complete() и embed()."""

    @patch("lightrag.llm.gemini.gemini_model_complete", new_callable=AsyncMock)
    async def test_complete_calls_gemini(self, mock_complete):
        """complete() вызывает gemini_model_complete с правильными параметрами."""
        mock_complete.return_value = "response"
        settings = _make_llm_settings()
        provider = GeminiProvider(settings)

        result = await provider.complete("test prompt", system_prompt="sys")

        assert result == "response"
        mock_complete.assert_awaited_once()
        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["api_key"] == "test-api-key"
        assert call_kwargs["model_name"] == "gemini-2.5-flash"
        assert call_kwargs["system_prompt"] == "sys"

    @patch("lightrag.llm.gemini.gemini_model_complete", new_callable=AsyncMock)
    async def test_complete_default_history_is_empty_list(self, mock_complete):
        """Без history_messages передаётся пустой список в Gemini."""
        mock_complete.return_value = "ok"
        provider = GeminiProvider(_make_llm_settings())

        await provider.complete("prompt")

        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["history_messages"] == []

    @patch("lightrag.llm.gemini.gemini_model_complete", new_callable=AsyncMock)
    async def test_complete_passes_extra_kwargs(self, mock_complete):
        """Дополнительные kwargs пробрасываются в gemini_model_complete."""
        mock_complete.return_value = "ok"
        provider = GeminiProvider(_make_llm_settings())

        await provider.complete("prompt", temperature=0.5, top_k=40)

        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_k"] == 40

    @patch("lightrag.llm.gemini.gemini_embed")
    async def test_embed_calls_gemini(self, mock_embed):
        """embed() вызывает gemini_embed.func с правильными параметрами."""
        expected = np.array([[0.1, 0.2, 0.3]])
        mock_embed.func = AsyncMock(return_value=expected)

        provider = GeminiProvider(_make_llm_settings())
        result = await provider.embed(["hello"])

        np.testing.assert_array_equal(result, expected)
        mock_embed.func.assert_awaited_once()
        call_kwargs = mock_embed.func.call_args[1]
        assert call_kwargs["api_key"] == "test-api-key"
        assert call_kwargs["model"] == "gemini-embedding-001"
        assert call_kwargs["embedding_dim"] == 768

    @patch("lightrag.llm.gemini.gemini_embed")
    async def test_embed_passes_texts(self, mock_embed):
        """Список текстов передаётся как первый аргумент."""
        mock_embed.func = AsyncMock(return_value=np.array([[0.0]]))
        provider = GeminiProvider(_make_llm_settings())

        texts = ["text1", "text2", "text3"]
        await provider.embed(texts)

        call_args = mock_embed.func.call_args[0]
        assert call_args[0] == texts


class TestLLMProviderProperties:
    """Тесты свойств LLMProvider."""

    def test_model_name(self):
        provider = GeminiProvider(_make_llm_settings(model_name="custom-model"))
        assert provider.model_name == "custom-model"

    def test_embedding_model(self):
        provider = GeminiProvider(_make_llm_settings(embedding_model="custom-embed"))
        assert provider.embedding_model == "custom-embed"

    def test_embedding_dim(self):
        provider = GeminiProvider(_make_llm_settings(embedding_dim=1024))
        assert provider.embedding_dim == 1024

    def test_max_token_size(self):
        provider = GeminiProvider(_make_llm_settings(max_token_size=4096))
        assert provider.max_token_size == 4096


@pytest.mark.asyncio
class TestOllamaProvider:
    """Тесты OllamaProvider: complete() и embed()."""

    @patch("lightrag.llm.ollama.ollama_model_complete", new_callable=AsyncMock)
    async def test_complete_calls_ollama(self, mock_complete):
        """complete() вызывает ollama_model_complete с правильными параметрами."""
        mock_complete.return_value = "response"
        provider = OllamaProvider(
            _make_llm_settings(
                provider="ollama",
                model_name="qwen2.5:7b",
                embedding_model="bge-m3:latest",
                ollama_host="http://ollama:11434",
            )
        )

        result = await provider.complete("test prompt", system_prompt="sys")

        assert result == "response"
        mock_complete.assert_awaited_once()
        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["system_prompt"] == "sys"
        assert call_kwargs["history_messages"] == []
        assert call_kwargs["host"] == "http://ollama:11434"
        assert call_kwargs["api_key"] == "test-api-key"

    @patch("lightrag.llm.ollama.ollama_embed")
    async def test_embed_calls_ollama(self, mock_embed):
        """embed() вызывает ollama_embed.func с правильными параметрами."""
        expected = np.array([[0.1, 0.2, 0.3]])
        mock_embed.func = AsyncMock(return_value=expected)

        provider = OllamaProvider(
            _make_llm_settings(
                provider="ollama",
                embedding_model="bge-m3:latest",
                ollama_host="http://ollama:11434",
            )
        )
        result = await provider.embed(["hello"])

        np.testing.assert_array_equal(result, expected)
        mock_embed.func.assert_awaited_once()
        call_kwargs = mock_embed.func.call_args[1]
        assert call_kwargs["embed_model"] == "bge-m3:latest"
        assert call_kwargs["embedding_dim"] == 768
        assert call_kwargs["host"] == "http://ollama:11434"


class TestCreateProvider:
    """Тесты фабрики create_provider."""

    def test_creates_gemini(self):
        """provider='gemini' → GeminiProvider."""
        settings = _make_llm_settings(provider="gemini")
        provider = create_provider(settings)
        assert isinstance(provider, GeminiProvider)

    def test_creates_ollama(self):
        """provider='ollama' → OllamaProvider."""
        settings = _make_llm_settings(provider="ollama")
        provider = create_provider(settings)
        assert isinstance(provider, OllamaProvider)

    def test_unknown_provider_raises(self):
        """Неизвестный провайдер — ValueError."""
        settings = _make_llm_settings()
        # Подменяем provider через model_copy
        bad_settings = settings.model_copy(update={"provider": "unknown"})
        with pytest.raises(ValueError, match="Неизвестный LLM-провайдер"):
            create_provider(bad_settings)

    def test_provider_receives_settings(self):
        """Провайдер получает настройки из LLMSettings."""
        settings = _make_llm_settings(model_name="my-model")
        provider = create_provider(settings)
        assert provider.model_name == "my-model"


class TestLLMSettingsValidation:
    """Валидация настроек LLM для разных провайдеров."""

    def test_gemini_requires_api_key(self):
        with pytest.raises(ValueError, match="LLM__API_KEY"):
            LLMSettings(provider="gemini", api_key=None)

    def test_ollama_allows_missing_api_key(self):
        settings = LLMSettings(provider="ollama", api_key=None)
        assert settings.provider == "ollama"


# ── RagService._llm_func ─────────────────────────────────


@pytest.mark.asyncio
class TestRagServiceLLMFunc:
    """Тесты RagService._llm_func: rate limiting + retry."""

    async def test_successful_call(self):
        """Успешный вызов — provider.complete вызывается один раз."""
        service = _make_service()

        result = await service._llm_func("test prompt")

        assert result == "LLM response"
        service._provider.complete.assert_awaited_once()
        service._wait_llm_rate.assert_awaited()

    async def test_passes_all_parameters(self):
        """Все параметры передаются в provider.complete."""
        service = _make_service()

        await service._llm_func(
            "prompt",
            system_prompt="system",
            history_messages=[{"role": "user", "content": "hi"}],
            keyword_extraction=True,
        )

        call_kwargs = service._provider.complete.call_args[1]
        assert call_kwargs["system_prompt"] == "system"
        assert call_kwargs["history_messages"] == [{"role": "user", "content": "hi"}]
        assert call_kwargs["keyword_extraction"] is True

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_429(self, mock_sleep):
        """429 ошибка → retry с задержкой, затем успех."""
        service = _make_service()
        service._provider.complete.side_effect = [
            Exception("429 Resource Exhausted"),
            "success after retry",
        ]

        result = await service._llm_func("prompt")

        assert result == "success after retry"
        assert service._provider.complete.await_count == 2
        mock_sleep.assert_awaited()

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_resource_exhausted(self, mock_sleep):
        """RESOURCE_EXHAUSTED → retry."""
        service = _make_service()
        service._provider.complete.side_effect = [
            Exception("RESOURCE_EXHAUSTED"),
            "ok",
        ]

        result = await service._llm_func("prompt")
        assert result == "ok"
        assert service._provider.complete.await_count == 2

    async def test_non_429_error_raises_immediately(self):
        """Не-429 ошибка поднимается сразу без retry."""
        service = _make_service()
        service._provider.complete.side_effect = ValueError("invalid model")

        with pytest.raises(ValueError, match="invalid model"):
            await service._llm_func("prompt")

        assert service._provider.complete.await_count == 1

    @patch("src.modules.rag.services.random.uniform", return_value=0.0)
    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_parses_retry_delay(self, mock_sleep, _mock_rand):
        """retryDelay в сообщении ошибки парсится и используется для ожидания."""
        service = _make_service()
        service._provider.complete.side_effect = [
            Exception("429 retryDelay: 10 seconds"),
            "ok",
        ]

        result = await service._llm_func("prompt")
        assert result == "ok"
        # sleep вызван с parsed delay (10 + 2 = 12.0)
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg == 12.0

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_max_retries_then_last_attempt(self, mock_sleep):
        """5 retry-ошибок 429 → финальная (6-я) попытка без перехвата."""
        service = _make_service()
        service._provider.complete.side_effect = [
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            "final success",
        ]

        result = await service._llm_func("prompt")
        assert result == "final success"
        assert service._provider.complete.await_count == 6  # 5 retries + 1 final

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_all_retries_exhausted_raises(self, mock_sleep):
        """Все попытки исчерпаны → последняя ошибка поднимается."""
        service = _make_service()
        service._provider.complete.side_effect = Exception("429 limit exceeded")

        with pytest.raises(Exception, match="429"):
            await service._llm_func("prompt")

        # 5 в цикле + 1 финальная = 6
        assert service._provider.complete.await_count == 6

    @patch("src.modules.rag.services.random.uniform", return_value=0.0)
    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_wait_capped_at_max_delay(self, mock_sleep, _mock_rand):
        """Задержка при retry ограничена llm_retry_max_delay (90с)."""
        service = _make_service()
        service._provider.complete.side_effect = [
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            "ok",
        ]

        await service._llm_func("prompt")

        # base_delay=15, formula: min(15 * 2^(attempt-1), 90)
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_calls[0] == 15.0   # attempt 1: 15*1
        assert sleep_calls[1] == 30.0   # attempt 2: 15*2
        assert sleep_calls[2] == 60.0   # attempt 3: 15*4
        assert sleep_calls[3] == 90.0   # attempt 4: min(15*8, 90)
        assert sleep_calls[4] == 90.0   # attempt 5: min(15*16, 90)


# ── RagService._embed_func ───────────────────────────────


@pytest.mark.asyncio
class TestRagServiceEmbedFunc:
    """Тесты RagService._embed_func: rate limiting + retry."""

    async def test_successful_call(self):
        """Успешный вызов возвращает numpy массив."""
        service = _make_service()
        expected = np.array([[0.1, 0.2, 0.3]])
        service._provider.embed.return_value = expected

        result = await service._embed_func(["test text"])

        np.testing.assert_array_equal(result, expected)
        service._provider.embed.assert_awaited_once()

    async def test_passes_texts_list(self):
        """Список текстов передаётся в provider.embed."""
        service = _make_service()
        texts = ["text1", "text2", "text3"]

        await service._embed_func(texts)

        service._provider.embed.assert_awaited_once_with(texts)

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_429(self, mock_sleep):
        """429 ошибка → retry, затем успех."""
        service = _make_service()
        service._provider.embed.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED"),
            np.array([[0.5, 0.5]]),
        ]

        result = await service._embed_func(["text"])

        np.testing.assert_array_equal(result, np.array([[0.5, 0.5]]))
        assert service._provider.embed.await_count == 2
        mock_sleep.assert_awaited()

    async def test_non_429_error_raises_immediately(self):
        """Не-429 ошибка поднимается без retry."""
        service = _make_service()
        service._provider.embed.side_effect = ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await service._embed_func(["text"])

        assert service._provider.embed.await_count == 1

    @patch("src.modules.rag.services.random.uniform", return_value=0.0)
    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_parses_retry_delay(self, mock_sleep, _mock_rand):
        """retryDelay в ошибке парсится."""
        service = _make_service()
        service._provider.embed.side_effect = [
            Exception("429 retryDelay: 20 seconds"),
            np.array([[0.0]]),
        ]

        await service._embed_func(["text"])

        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg == 22.0  # 20 + 2

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_all_retries_exhausted_raises(self, mock_sleep):
        """Все 5 retry + финальная попытка → исключение."""
        service = _make_service()
        service._provider.embed.side_effect = Exception("429")

        with pytest.raises(Exception, match="429"):
            await service._embed_func(["text"])

        # 5 в цикле + 1 финальная = 6
        assert service._provider.embed.await_count == 6

    @patch("src.modules.rag.services.random.uniform", return_value=0.0)
    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_wait_capped_at_max_delay(self, mock_sleep, _mock_rand):
        """Задержка embedding retry ограничена embed_retry_max_delay (120с)."""
        service = _make_service()
        service._provider.embed.side_effect = [
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            np.array([[0.0]]),
        ]

        await service._embed_func(["text"])

        # base_delay=30, formula: min(30 * 2^(attempt-1), 120)
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_calls[0] == 30.0   # attempt 1: 30*1
        assert sleep_calls[1] == 60.0   # attempt 2: 30*2
        assert sleep_calls[2] == 120.0  # attempt 3: 30*4
        assert sleep_calls[3] == 120.0  # attempt 4: min(30*8, 120)
        assert sleep_calls[4] == 120.0  # attempt 5: min(30*16, 120)

    async def test_multiple_texts_embedding(self):
        """Батч из нескольких текстов обрабатывается как один вызов."""
        service = _make_service()
        expected = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        service._provider.embed.return_value = expected

        result = await service._embed_func(["text1", "text2", "text3"])

        np.testing.assert_array_equal(result, expected)
        service._provider.embed.assert_awaited_once()


# ── _build_embedding_func ────────────────────────────────


@pytest.mark.asyncio
class TestBuildEmbeddingFunc:
    """Тесты RagService._build_embedding_func."""

    async def test_returns_embedding_func(self):
        """Возвращает EmbeddingFunc с правильными атрибутами."""
        from lightrag.utils import EmbeddingFunc

        service = _make_service()
        func = service._build_embedding_func()

        assert isinstance(func, EmbeddingFunc)

    async def test_embedding_dim_from_provider(self):
        """embedding_dim берётся из провайдера."""
        provider = _make_mock_provider()
        provider.embedding_dim = 1024
        service = _make_service(provider)

        func = service._build_embedding_func()
        assert func.embedding_dim == 1024

    async def test_max_token_size_from_provider(self):
        """max_token_size берётся из провайдера."""
        provider = _make_mock_provider()
        provider.max_token_size = 4096
        service = _make_service(provider)

        func = service._build_embedding_func()
        assert func.max_token_size == 4096

    async def test_func_is_callable(self):
        """Внутренняя функция — callable (bound method _embed_func)."""
        service = _make_service()
        func = service._build_embedding_func()
        assert callable(func.func)


# ── Singleton ─────────────────────────────────────────────


@pytest.mark.asyncio
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


# ── Edge Cases ────────────────────────────────────────────


@pytest.mark.asyncio
class TestRagServiceWrapperEdgeCases:
    """Дополнительные edge-case тесты для LLM/Embed wrappers."""

    async def test_empty_prompt(self):
        """Пустой промпт всё равно передаётся."""
        service = _make_service()
        service._provider.complete.return_value = ""

        result = await service._llm_func("")
        assert result == ""
        assert service._provider.complete.call_args[0][0] == ""

    async def test_long_prompt(self):
        """Длинный промпт передаётся без обрезки."""
        service = _make_service()
        long_prompt = "x" * 100_000
        service._provider.complete.return_value = "ok"

        await service._llm_func(long_prompt)
        assert service._provider.complete.call_args[0][0] == long_prompt

    async def test_unicode_prompt(self):
        """Unicode-промпт на разных языках."""
        service = _make_service()
        prompt = "Привет 你好 مرحبا"
        service._provider.complete.return_value = "ответ"

        result = await service._llm_func(prompt)
        assert result == "ответ"
        assert service._provider.complete.call_args[0][0] == prompt

    async def test_extra_kwargs_passed_through(self):
        """Дополнительные kwargs пробрасываются в provider.complete."""
        service = _make_service()
        service._provider.complete.return_value = "ok"

        await service._llm_func("prompt", temperature=0.5, top_k=40)

        call_kwargs = service._provider.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_k"] == 40

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_429_then_non_429_error(self, mock_sleep):
        """Сначала 429, потом другая ошибка → вторая ошибка поднимается."""
        service = _make_service()
        service._provider.complete.side_effect = [
            Exception("429"),
            TypeError("wrong type"),
        ]

        with pytest.raises(TypeError, match="wrong type"):
            await service._llm_func("prompt")

        assert service._provider.complete.await_count == 2

    @patch("src.modules.rag.services.asyncio.sleep", new_callable=AsyncMock)
    async def test_embed_429_then_other_error(self, mock_sleep):
        """Embedding: 429, потом другая ошибка."""
        service = _make_service()
        service._provider.embed.side_effect = [
            Exception("RESOURCE_EXHAUSTED"),
            RuntimeError("connection lost"),
        ]

        with pytest.raises(RuntimeError, match="connection lost"):
            await service._embed_func(["text"])

        assert service._provider.embed.await_count == 2
