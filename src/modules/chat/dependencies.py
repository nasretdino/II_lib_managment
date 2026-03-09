from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.modules.agents.dependencies import get_agents_service
from src.modules.agents.services import AgentsService

from .dao import ChatDAO
from .services import ChatService


async def get_chat_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    agents: Annotated[AgentsService, Depends(get_agents_service)],
) -> ChatService:
    return ChatService(ChatDAO(db), agents)
