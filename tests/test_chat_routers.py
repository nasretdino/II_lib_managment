import pytest
from httpx import AsyncClient

from src.main import app
from src.modules.agents.dependencies import get_agents_service
from src.modules.agents.schemas import AgentEventRead, AgentRunResponse


pytestmark = pytest.mark.asyncio


class FakeAgentsService:
    async def run(self, question: str, conversation_history: list[dict[str, str]], max_iterations: int = 3):
        return AgentRunResponse(
            question=question,
            query_type="search",
            retrieval_mode="local",
            final_answer="Chat router streamed response",
            sources=["s1"],
            iterations=1,
            is_approved=True,
            events=[
                AgentEventRead(type="ask_routed", data="route", iteration=0),
                AgentEventRead(type="retrieve_completed", data="retrieve", iteration=0),
                AgentEventRead(type="emit_completed", data="emit", iteration=1),
            ],
        )


@pytest.fixture(autouse=True)
def override_agents_dependency():
    app.dependency_overrides[get_agents_service] = lambda: FakeAgentsService()
    yield
    app.dependency_overrides.pop(get_agents_service, None)


async def _create_user(client: AsyncClient) -> int:
    response = await client.post("/users/", json={"name": "ChatApiUser"})
    return response.json()["id"]


async def test_chat_stream_and_history(client: AsyncClient):
    user_id = await _create_user(client)

    async with client.stream(
        "POST",
        "/chat/stream",
        json={"user_id": user_id, "message": "What is Python?"},
    ) as response:
        assert response.status_code == 200
        body = ""
        async for chunk in response.aiter_text():
            body += chunk

    assert "event: routing" in body
    assert "event: token" in body
    assert "event: done" in body

    sessions_resp = await client.get("/chat/sessions", params={"user_id": user_id})
    assert sessions_resp.status_code == 200
    sessions = sessions_resp.json()
    assert len(sessions) == 1
    assert sessions[0]["message_count"] == 2

    session_id = sessions[0]["id"]
    messages_resp = await client.get(f"/chat/sessions/{session_id}/messages")
    assert messages_resp.status_code == 200
    messages = messages_resp.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


async def test_chat_stream_invalid_session_returns_404(client: AsyncClient):
    user_id = await _create_user(client)
    response = await client.post(
        "/chat/stream",
        json={"user_id": user_id, "session_id": 9999, "message": "Hello"},
    )
    assert response.status_code == 404


async def test_chat_messages_not_found(client: AsyncClient):
    response = await client.get("/chat/sessions/9999/messages")
    assert response.status_code == 404
