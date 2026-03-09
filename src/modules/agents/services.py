from src.core import get_logger
from src.modules.agents.schemas import AgentEventRead, AgentRunResponse
from src.modules.agents.state import AgentState
from src.modules.agents.workflow import build_workflow
from src.modules.rag.services import RagService


logger = get_logger(module="agents", component="service")


class AgentsService:
    def __init__(self, rag: RagService):
        logger.info("Initializing AgentsService")
        self._workflow = build_workflow(rag)

    async def run(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        max_iterations: int = 3,
    ) -> AgentRunResponse:
        logger.info(
            "Agents pipeline started: question_len={}, history_items={}, max_iterations={}",
            len(question),
            len(conversation_history or []),
            max_iterations,
        )
        initial_state: AgentState = {
            "question": question,
            "conversation_history": conversation_history or [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "is_approved": False,
            "needs_more_context": False,
            "events": [],
            "sources": [],
            "retrieval_mode": "hybrid",
            "raw": {},
        }

        try:
            final_state = await self._workflow.ainvoke(initial_state)
        except Exception:
            logger.exception("Agents pipeline failed")
            raise

        events = [AgentEventRead(**item) for item in final_state.get("events", [])]

        logger.info(
            "Agents pipeline completed: query_type={}, iterations={}, approved={}, events={}",
            final_state.get("query_type", "search"),
            final_state.get("iteration", 0),
            final_state.get("is_approved", False),
            len(events),
        )

        return AgentRunResponse(
            question=question,
            query_type=final_state.get("query_type", "search"),
            retrieval_mode=final_state.get("retrieval_mode", "hybrid"),
            final_answer=final_state.get("final_answer", ""),
            sources=final_state.get("sources", []),
            iterations=final_state.get("iteration", 0),
            is_approved=final_state.get("is_approved", False),
            events=events,
        )
