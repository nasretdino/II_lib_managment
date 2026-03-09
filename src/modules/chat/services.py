import asyncio
import re
from collections.abc import AsyncGenerator

from src.core import NotFoundError, get_logger
from src.modules.agents.services import AgentsService

from .dao import ChatDAO
from .models import ChatMessage, ChatSession
from .schemas import ChatEvent, ChatRequest, MessageRead, SessionRead


logger = get_logger(module="chat", component="service")


class ChatService:
    def __init__(self, dao: ChatDAO, agents: AgentsService):
        self._dao = dao
        self._agents = agents

    async def prepare_session(self, request: ChatRequest) -> ChatSession:
        if request.session_id is None:
            session = await self._dao.create_session(user_id=request.user_id)
            logger.info("Created chat session id={} for user_id={}", session.id, request.user_id)
            return session

        session = await self._dao.find_session_for_user(request.session_id, request.user_id)
        if session is None:
            raise NotFoundError("Chat session not found")
        return session

    async def get_sessions(self, user_id: int) -> list[SessionRead]:
        rows = await self._dao.get_sessions_by_user(user_id)
        return [
            SessionRead(
                id=session.id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=message_count,
            )
            for session, message_count in rows
        ]

    async def get_messages(self, session_id: int) -> list[MessageRead]:
        session = await self._dao.find_one_or_none_by_id(session_id)
        if session is None:
            raise NotFoundError("Chat session not found")

        messages = await self._dao.get_messages_by_session(session_id)
        return [MessageRead.model_validate(message) for message in messages]

    async def stream_response(
        self,
        request: ChatRequest,
        session_id: int,
    ) -> AsyncGenerator[ChatEvent, None]:
        user_message = await self._dao.add_message(
            session_id=session_id,
            role="user",
            content=request.message,
        )
        await self._dao.set_session_title_if_empty(
            session_id=session_id,
            title=self._build_title(request.message),
        )

        history_messages = await self._dao.get_messages_by_session(session_id)
        history = self._build_history(history_messages, skip_message_id=user_message.id)

        yield ChatEvent(event="routing", data="Определяю тип запроса...")
        yield ChatEvent(event="retrieving", data="Ищу данные в базе знаний...")

        run_task = asyncio.create_task(
            self._agents.run(
                question=request.message,
                conversation_history=history,
            )
        )

        try:
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(run_task), timeout=15)
                    break
                except TimeoutError:
                    yield ChatEvent(event="keepalive", data="")

            for agent_event in result.events:
                yield ChatEvent(
                    event=self._map_agent_event(agent_event.type),
                    data=agent_event.data,
                )

            for token in self._iterate_tokens(result.final_answer):
                yield ChatEvent(event="token", data=token)

            await self._dao.add_message(
                session_id=session_id,
                role="assistant",
                content=result.final_answer,
                metadata={
                    "query_type": result.query_type,
                    "retrieval_mode": result.retrieval_mode,
                    "sources": result.sources,
                    "iterations": result.iterations,
                    "is_approved": result.is_approved,
                },
            )

            yield ChatEvent(event="done", data="")
        except Exception:
            logger.exception("Chat stream failed: session_id={} user_id={}", session_id, request.user_id)
            yield ChatEvent(event="error", data="Не удалось сгенерировать ответ")

    @staticmethod
    def _build_title(message: str) -> str:
        normalized = " ".join(message.strip().split())
        if len(normalized) <= 60:
            return normalized
        return normalized[:57] + "..."

    @staticmethod
    def _build_history(
        messages: list[ChatMessage],
        skip_message_id: int | None = None,
    ) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for message in messages:
            if skip_message_id is not None and message.id == skip_message_id:
                continue
            if message.role not in {"user", "assistant", "system"}:
                continue
            history.append({"role": message.role, "content": message.content})
        return history

    @staticmethod
    def _iterate_tokens(text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"\S+\s*", text)

    @staticmethod
    def _map_agent_event(agent_event_type: str) -> str:
        mapping = {
            "ask_routed": "routing",
            "retrieve_completed": "retrieving",
            "analyze_completed": "analyst_thinking",
            "critique_completed": "critic_review",
            "decision_made": "decision",
            "emit_completed": "emit",
        }
        return mapping.get(agent_event_type, "status")
