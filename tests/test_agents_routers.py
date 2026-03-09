import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


async def test_agents_run_endpoint(client: AsyncClient):
    resp = await client.post(
        "/agents/run",
        json={
            "question": "What is Python?",
            "conversation_history": [],
            "max_iterations": 3,
        },
    )

    assert resp.status_code == 200
    data = resp.json()

    assert data["question"] == "What is Python?"
    assert data["query_type"] in {"search", "analytics", "podcast"}
    assert isinstance(data["final_answer"], str)
    assert isinstance(data["events"], list)


async def test_agents_run_validates_empty_question(client: AsyncClient):
    resp = await client.post("/agents/run", json={"question": ""})
    assert resp.status_code == 422
