from src.core import get_logger
from src.modules.rag.schemas import SearchResult
from src.modules.rag.services import RagService


logger = get_logger(module="agents", component="tools")


class AgentTools:
    """Tool layer used by the researcher node."""

    def __init__(self, rag: RagService):
        self._rag = rag

    async def search_knowledge(
        self,
        query: str,
        mode: str = "hybrid",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> SearchResult:
        logger.debug(
            "RAG search requested: mode={}, query_len={}, history_items={}",
            mode,
            len(query),
            len(conversation_history or []),
        )
        result = await self._rag.search(
            query=query,
            mode=mode,
            conversation_history=conversation_history,
            only_context=True,
        )
        logger.info(
            "RAG search completed: mode={}, context_len={}, sources={}",
            mode,
            len(result.context_text),
            len(result.sources),
        )
        return result
