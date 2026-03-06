"""
Кастомные исключения приложения.

Перехватываются через exception_handler в main.py
и превращаются в корректные HTTP-ответы.
"""


class NotFoundError(Exception):
    """Ресурс не найден (→ 404)."""

    def __init__(self, detail: str = "Resource not found"):
        self.detail = detail
        super().__init__(detail)


class ConflictError(Exception):
    """Конфликт данных, например дубликат (→ 409)."""

    def __init__(self, detail: str = "Conflict"):
        self.detail = detail
        super().__init__(detail)


class DocumentParsingError(Exception):
    """Ошибка парсинга документа (→ 422)."""

    def __init__(self, detail: str = "Failed to parse document"):
        self.detail = detail
        super().__init__(detail)


class LLMProviderError(Exception):
    """Ошибка обращения к LLM-провайдеру (→ 502)."""

    def __init__(self, detail: str = "LLM provider error"):
        self.detail = detail
        super().__init__(detail)
