from unittest.mock import AsyncMock

import pytest

from src.modules.agents.services import AgentsService
from src.modules.agents.state import AgentState
from src.modules.agents.workflow import build_workflow
from src.modules.rag.schemas import SearchResult


pytestmark = pytest.mark.asyncio


class FakeRag:
    def __init__(self):
        self.search = AsyncMock(
            return_value=SearchResult(
                context_text="[doc_id=1] Python is a programming language.",
                sources=["1"],
                mode="hybrid",
            )
        )


async def test_agents_service_runs_full_arcade_cycle():
    rag = FakeRag()
    service = AgentsService(rag)

    result = await service.run(question="What is Python?")

    assert result.query_type == "search"
    assert result.retrieval_mode in {"local", "hybrid", "global"}
    assert isinstance(result.final_answer, str)
    assert result.final_answer
    assert any(event.type == "emit_completed" for event in result.events)


async def test_agents_service_routes_analytics_query():
    rag = FakeRag()
    service = AgentsService(rag)

    result = await service.run(question="Compare all documents and show differences")

    assert result.query_type == "analytics"


async def test_workflow_decision_path_end_when_approved(monkeypatch):
    rag = FakeRag()
    workflow = build_workflow(rag)

    initial_state: AgentState = {
        "question": "What is Python?",
        "conversation_history": [],
        "iteration": 0,
        "max_iterations": 2,
        "is_approved": False,
        "needs_more_context": False,
        "events": [],
        "sources": [],
        "retrieval_mode": "hybrid",
    }

    final_state = await workflow.ainvoke(initial_state)

    assert final_state.get("route_after_decision") == "end"
    assert final_state.get("final_answer")
