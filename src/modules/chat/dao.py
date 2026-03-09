from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import func, select, update

from src.db import BaseDAO

from .models import ChatMessage, ChatSession


class ChatDAO(BaseDAO[ChatSession]):
    model = ChatSession

    async def create_session(self, user_id: int, title: str | None = None) -> ChatSession:
        return await self.add({"user_id": user_id, "title": title}, flush=True)

    async def find_session_for_user(self, session_id: int, user_id: int) -> ChatSession | None:
        records = await self.find_all(filters={"id": session_id, "user_id": user_id}, limit=1)
        return records[0] if records else None

    async def set_session_title_if_empty(self, session_id: int, title: str) -> None:
        session = await self.find_one_or_none_by_id(session_id)
        if session is None or session.title:
            return
        session.title = title
        session.updated_at = datetime.now(timezone.utc)

    async def touch_session(self, session_id: int) -> None:
        await self._session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(updated_at=datetime.now(timezone.utc))
        )

    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            metadata_=metadata,
        )
        self._session.add(message)
        await self._session.flush()
        await self._session.refresh(message)
        await self.touch_session(session_id)
        return message

    async def get_sessions_by_user(self, user_id: int) -> Sequence[tuple[ChatSession, int]]:
        stmt = (
            select(ChatSession, func.count(ChatMessage.id).label("message_count"))
            .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
            .where(ChatSession.user_id == user_id)
            .group_by(ChatSession.id)
            .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        )
        rows = await self._session.execute(stmt)
        return rows.all()

    async def get_messages_by_session(self, session_id: int) -> list[ChatMessage]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        )
        rows = await self._session.execute(stmt)
        return list(rows.scalars().all())
