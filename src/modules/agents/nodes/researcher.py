from src.core import get_logger
from src.modules.agents.state import AgentState, QueryType
from src.modules.agents.tools import AgentTools


logger = get_logger(module="agents", node="retrieve")


def _mode_for_query_type(query_type: QueryType) -> str:
    if query_type == "search":
        # mix combines graph and vector retrieval and works better for document Q&A.
        return "mix"
    if query_type == "analytics":
        return "hybrid"
    return "global"


def build_researcher_node(tools: AgentTools):
    async def researcher_node(state: AgentState) -> AgentState:
        mode = _mode_for_query_type(state["query_type"])
        logger.info(
            "Retrieve node started: mode={}, query_type={}, iteration={}",
            mode,
            state.get("query_type", "search"),
            state.get("iteration", 0),
        )

        question = state["question"]
        correction = state.get("correction")
        if correction:
            logger.debug("Retrieve node received correction, injecting into query")
            question = f"{question}\n\nCorrection from critic:\n{correction}"

        result = await tools.search_knowledge(
            query=question,
            mode=mode,
            conversation_history=state.get("conversation_history", []),
        )
        logger.info(
            "Retrieve node completed: context_len={}, sources={}",
            len(result.context_text),
            len(result.sources),
        )

        events = list(state.get("events", []))
        events.append(
            {
                "type": "retrieve_completed",
                "data": f"Retrieved context with mode={mode}, sources={len(result.sources)}",
                "iteration": state.get("iteration", 0),
            }
        )

        return {
            **state,
            "retrieval_mode": mode,
            "context": result.context_text,
            "sources": result.sources,
            "events": events,
        }

    return researcher_node
