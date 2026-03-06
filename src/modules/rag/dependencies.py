"""FastAPI Depends для RAG-модуля."""

from .services import RagService, get_rag_service


def get_rag() -> RagService:
    """DI-зависимость: получить инициализированный RagService."""
    return get_rag_service()
