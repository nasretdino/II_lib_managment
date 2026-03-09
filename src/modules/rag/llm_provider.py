"""
Абстракция LLM-провайдера для RAG-модуля.

Позволяет подключать разные LLM-провайдеры (Gemini, OpenAI, etc.)
без изменения бизнес-логики RagService.

Добавление нового провайдера:
  1. Создать класс-наследник LLMProvider
  2. Зарегистрировать в _PROVIDERS
  3. Добавить Literal-значение в LLMSettings.provider (config.py)
"""

from abc import ABC, abstractmethod

import numpy as np

from src.core.config import LLMSettings


class LLMProvider(ABC):
    """
    Абстрактный LLM-провайдер: text completion + embeddings.

    Каждый провайдер инкапсулирует работу с конкретным API
    (Gemini, OpenAI, Anthropic, …), предоставляя единый интерфейс
    для RagService.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def embedding_model(self) -> str:
        return self._settings.embedding_model

    @property
    def embedding_dim(self) -> int:
        return self._settings.embedding_dim

    @property
    def max_token_size(self) -> int:
        return self._settings.max_token_size

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        """Генерация текста через LLM."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Создание эмбеддингов для списка текстов."""
        ...


class GeminiProvider(LLMProvider):
    """LLM-провайдер на базе Google Gemini."""

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        from lightrag.llm.gemini import gemini_model_complete

        if self._settings.api_key is None:
            raise ValueError("Для Gemini требуется LLM__API_KEY")

        return await gemini_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            keyword_extraction=keyword_extraction,
            api_key=self._settings.api_key.get_secret_value(),
            model_name=self._settings.model_name,
            **kwargs,
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        from lightrag.llm.gemini import gemini_embed

        if self._settings.api_key is None:
            raise ValueError("Для Gemini требуется LLM__API_KEY")

        return await gemini_embed.func(
            texts,
            api_key=self._settings.api_key.get_secret_value(),
            model=self._settings.embedding_model,
            embedding_dim=self._settings.embedding_dim,
        )


class OllamaProvider(LLMProvider):
    """LLM-провайдер на базе Ollama."""

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        from lightrag.llm.ollama import ollama_model_complete

        return await ollama_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            keyword_extraction=keyword_extraction,
            host=self._settings.ollama_host,
            api_key=(
                self._settings.api_key.get_secret_value()
                if self._settings.api_key is not None
                else None
            ),
            **kwargs,
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        from lightrag.llm.ollama import ollama_embed

        return await ollama_embed.func(
            texts,
            embed_model=self._settings.embedding_model,
            embedding_dim=self._settings.embedding_dim,
            host=self._settings.ollama_host,
            api_key=(
                self._settings.api_key.get_secret_value()
                if self._settings.api_key is not None
                else None
            ),
        )


# ── Реестр провайдеров ────────────────────────────────────

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


def create_provider(settings: LLMSettings) -> LLMProvider:
    """Фабрика: создать LLM-провайдер по настройкам."""
    provider_cls = _PROVIDERS.get(settings.provider)
    if provider_cls is None:
        raise ValueError(
            f"Неизвестный LLM-провайдер: {settings.provider!r}. "
            f"Доступные: {', '.join(_PROVIDERS)}"
        )
    return provider_cls(settings)
