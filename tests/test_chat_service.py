import pytest

from src.core.exceptions import NotFoundError
from src.modules.agents.schemas import AgentEventRead, AgentRunResponse
from src.modules.chat.dao import ChatDAO
from src.modules.chat.schemas import ChatRequest
from src.modules.chat.services import ChatService
from src.modules.users.dao import UserDAO
from src.modules.users.schemas import UserCreate


pytestmark = pytest.mark.asyncio


class FakeAgentsService:
    async def run(self, question: str, conversation_history: list[dict[str, str]], max_iterations: int = 3):
        return AgentRunResponse(
            question=question,
            query_type="search",
            retrieval_mode="local",
            final_answer="This is a streamed answer.",
            sources=["doc-1"],
            iterations=1,
            is_approved=True,
            events=[
                AgentEventRead(type="ask_routed", data="routed", iteration=0),
                AgentEventRead(type="emit_completed", data="ready", iteration=1),
            ],
        )


@pytest.fixture
async def user(db_session):
    dao = UserDAO(db_session)
    return await dao.add(UserCreate(name="ServiceUser"), flush=True)


@pytest.fixture
def chat_service(db_session):
    return ChatService(ChatDAO(db_session), FakeAgentsService())


class TestChatService:
    async def test_prepare_session_creates_new(self, chat_service: ChatService, user):
        req = ChatRequest(user_id=user.id, message="Hello")
        session = await chat_service.prepare_session(req)
        assert session.id is not None

    async def test_prepare_session_not_found(self, chat_service: ChatService, user):
        req = ChatRequest(user_id=user.id, session_id=9999, message="Hello")
        with pytest.raises(NotFoundError):
            await chat_service.prepare_session(req)

    async def test_stream_response_saves_messages(self, chat_service: ChatService, user):
        req = ChatRequest(user_id=user.id, message="What is Python?")
        session = await chat_service.prepare_session(req)

        events = [event async for event in chat_service.stream_response(req, session.id)]
        event_names = [event.event for event in events]

        assert "token" in event_names
        assert event_names[-1] == "done"

        messages = await chat_service.get_messages(session.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    async def test_get_sessions_returns_message_count(self, chat_service: ChatService, user):
        req = ChatRequest(user_id=user.id, message="First")
        session = await chat_service.prepare_session(req)
        _ = [event async for event in chat_service.stream_response(req, session.id)]

        sessions = await chat_service.get_sessions(user.id)
        assert len(sessions) == 1
        assert sessions[0].message_count == 2
