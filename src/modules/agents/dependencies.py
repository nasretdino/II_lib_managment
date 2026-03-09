from typing import Annotated

from fastapi import Depends

from src.modules.rag.dependencies import get_rag
from src.modules.rag.services import RagService

from .services import AgentsService


def get_agents_service(
    rag: Annotated[RagService, Depends(get_rag)],
) -> AgentsService:
    return AgentsService(rag)
